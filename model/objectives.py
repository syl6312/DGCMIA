import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm_label(image_fetures, text_fetures,  logit_scale=50.0, pid=None, image_id=None, factor=0.3, epsilon=1e-6):
    """
    Similarity Distribution Matching
    """

    labels = pid.half()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)


    logits_image_image = labels.where(labels!= 0, torch.tensor(float('-inf')))
    logit_image = F.softmax(logits_image_image, dim=1)
    # logit_image = labels / labels.sum(dim=1)
                
                    
    logits_cross = (logit_scale * text_norm @ image_norm.t() ) 
    logits_cross1 = (logit_scale * image_norm @ text_norm.t() ) 
    
    i2t_pred = F.softmax(logits_cross, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(logits_cross, dim=1) - torch.log(logit_image + epsilon))
    t2i_pred = F.softmax(logits_cross1, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(logits_cross1, dim=1) - torch.log(logit_image + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss




def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)



def compute_itc(image_features, text_features, logit_scale=50.0):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss





class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算锚点与正样本之间的欧氏距离
        pos_dist = F.pairwise_distance(anchor, positive)
        # 计算锚点与负样本之间的欧氏距离
        neg_dist = F.pairwise_distance(anchor, negative)

        # 按照三元组损失公式计算损失
        losses = torch.max(pos_dist - neg_dist + self.margin, torch.zeros_like(pos_dist))
        return losses.mean() 

def get_soft_labels(image_features, text_features):
    batch_size = text_features.shape[0]
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    
    with torch.no_grad():
        cosine_similarity_image = image_norm @ image_norm.t()  
        cosine_similarity_text = text_norm @ text_norm.t()  
        values, indices = torch.topk(cosine_similarity_image, 6, dim=1)
        values1, indices1 = torch.topk(cosine_similarity_text, 6, dim=1)
        mask = torch.zeros_like(cosine_similarity_image).scatter_(1, indices, 1)
        mask1 = torch.zeros_like(cosine_similarity_text).scatter_(1, indices1, 1)
        cosine_similarity_image = cosine_similarity_image * mask
        cosine_similarity_text = cosine_similarity_text * mask1        
        
        # Step 3: Generate pseudo-labels based on similarity threshold
        image_labels = (cosine_similarity_image > 0.7).float()
        text_labels = (cosine_similarity_text > 0.6).float()
        labels = (image_labels * text_labels) 
                
        batch1_labels = [-1] * labels.shape[0]
        current_class_id = 0

        # 遍历索引矩阵，进行分类标记
        for i in range(labels.shape[0]):
            if batch1_labels[i] == -1:
                # 为新类别的样本分配类别编号
                batch1_labels[i] = current_class_id
                # 找到与当前样本相似的其他样本，并标记为同一类别
                similar_indices = torch.where(labels[i] == 1)[0]
                for similar_idx in similar_indices:
                    batch1_labels[similar_idx] = current_class_id
                current_class_id += 1

        batch1_labels = torch.tensor(batch1_labels).to(image_features.device)
        unique_classes1, inverse_indices1 = torch.unique(batch1_labels, return_inverse=True)  

        pid = batch1_labels.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).half()
        # labels = (labels-torch.eye(batch_size).cuda()) * 0.5 + torch.eye(batch_size).cuda()
                


        unique_classes1, inverse_indices1 = torch.unique(batch1_labels, return_inverse=True)  
                
        element_counts = torch.bincount(inverse_indices1)

        result_indices = []
        for index in range(len(batch1_labels)):
            element = batch1_labels[index]
            element_indices = torch.where(batch1_labels == element)[0]
            if len(element_indices)>1:
                new_lst = find_index_in_tensor(element_indices, index)            
                if new_lst!= -1:
                    first_part = element_indices[0:new_lst]
                    second_part = element_indices[new_lst + 1:]
                    new_tensor = torch.cat([torch.tensor([index]).cuda(), first_part, second_part])
            else:
                new_tensor = element_indices         

            result_indices.append(new_tensor.tolist())
            
        row_sums = labels.sum(dim=1)
        valid_row_indices = torch.where(row_sums > 1)[0] 


        # result_indices = []
        # for element, count in zip(unique_classes1, element_counts):
        #     if count >= 1:
        #         element_indices = torch.where(batch1_labels == element)[0]
        #         result_indices.append(element_indices.tolist())
        #         # result_indices.extend(element_indices.tolist())   
                
        new_result_indices = []       
        for i in valid_row_indices:
            new_result_indices.append(result_indices[i])
                
    return result_indices, valid_row_indices 


def get_similarity_soft_base(image_features, text_features, thred1, thred2, logit_scale=50.0, metric='sdm', factor=0.3, epsilon=1e-6, logit_scale1=50.0):
    batch_size = text_features.shape[0]
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)
    # dropout = nn.Dropout(p=0.3)
    # text_norm = dropout(text_norm)
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    
    with torch.no_grad():
        cosine_similarity_image = image_norm @ image_norm.t()  
        cosine_similarity_text = text_norm @ text_norm.t()  
        
        image_labels = (cosine_similarity_image > thred1).float()
        text_labels = (cosine_similarity_text > thred2).float()
        labels = (image_labels * text_labels) 
        
        labels = (labels-torch.eye(batch_size).cuda()) * 0.1 + torch.eye(batch_size).cuda()
          
    if metric == 'sdm':          
        logits_image_image = labels.where(labels!= 0, torch.tensor(float('-inf')))
        
        logit_image = F.softmax(logits_image_image, dim=1)
                     
        logits_cross = (logit_scale * text_norm @ image_norm.t() ) 
        logits_cross1 = (logit_scale * image_norm @ text_norm.t() ) 
        
        i2t_pred = F.softmax(logits_cross, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(logits_cross, dim=1) - torch.log(logit_image + epsilon))
        t2i_pred = F.softmax(logits_cross1, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(logits_cross1, dim=1) - torch.log(logit_image + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        
        return loss 

    raise ValueError()    




def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-6):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

