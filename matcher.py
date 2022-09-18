import logging
from modules import *
import torch
import numpy as np

class EmbedMatcher(nn.Module):
    """
    基于嵌入的匹配模型   matching processor
    """

    def __init__(self, embed_dim, dropout=0.2, batch_size=64,
                 process_steps=4, aggregate='max', factors=6, ent_embeddings=None, rel_embeddings=None):
        super(EmbedMatcher, self).__init__()
        self.k = factors
        self.embed_dim = embed_dim

        self.aggregate = aggregate

        self.ent_embeddings = nn.Embedding(ent_embeddings.shape[0], self.embed_dim)
        self.ent_embeddings.weight.requires_grad = False

        self.rel_embeddings = nn.Embedding(rel_embeddings.shape[0], self.embed_dim)

        self.ent_embeddings.weight.data.copy_(ent_embeddings)
        self.rel_embeddings.weight.data.copy_(rel_embeddings)
        
	# nn.Linear() 全连接层
        # [batch_size,in_features] * [in_features,out_features] -> [batch_size,out_features]
        self.gcn_w = nn.Linear(self.embed_dim, self.embed_dim)
        init.xavier_normal_(self.gcn_w.weight)  # 全连接层的权重初始化
        init.constant_(self.gcn_w.bias, 0)  # 全连接层的偏置初始化

        self.dropout = nn.Dropout(0.5)

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)
        self.Bilinear = nn.Bilinear(embed_dim, embed_dim, 1, bias=False)

        self.fc1 = nn.Linear(2 * self.embed_dim, 1)
        init.xavier_normal_(self.fc1.weight)  # 全连接层的权重初始化
        init.constant_(self.fc1.bias, 0)  # 全连接层的偏置初始化
        self.fc2 = nn.Linear(2 * self.embed_dim, 1)
        init.xavier_normal_(self.fc2.weight)  # 全连接层的权重初始化
        init.constant_(self.fc2.bias, 0)  # 全连接层的偏置初始化

        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear3 = nn.Linear(self.embed_dim, self.embed_dim)
        init.xavier_normal_(self.linear1.weight)
        init.constant_(self.linear1.bias, 0)
        init.xavier_normal_(self.linear2.weight)
        init.constant_(self.linear2.bias, 0)
        init.xavier_normal_(self.linear3.weight)
        init.constant_(self.linear3.bias, 0)
        
        self.attention1 = Attention(200, 1, dropout)
        self.attention2 = Attention(200, 1, dropout)

    def neighbor_encoder(self, left_entity, right_entity, connections, num_neighbors, rel_embeds, ent_embeds):
        '''
        entity: 当前要编码的实体
        '''

        # [128, 50, 2] [128, 50, 2]
        connections_out, connections_in = connections

        # 维度变换 : [few/batch] -> [few/batch,1]
        num_neighbors = num_neighbors.unsqueeze(1)

        # [few/batch, max_neighbor]
        relations_out = connections_out[:, :, 0].squeeze(-1)
        entities_out = connections_out[:, :, 1].squeeze(-1)
        relations_in = connections_in[:, :, 0].squeeze(-1)
        entities_in = connections_in[:, :, 1].squeeze(-1)

        # [few/batch, max_neighbor, k, embed_dim]
        rel_embeds_out = self.dropout(rel_embeds[relations_out])
        ent_embeds_out = self.dropout(ent_embeds[entities_out])
        rel_embeds_in = self.dropout(rel_embeds[relations_in])
        ent_embeds_in = self.dropout(ent_embeds[entities_in])

        # [few/batch, k, embed_dim]
        left_ent_embeds = self.dropout(ent_embeds[left_entity])
        right_ent_embeds = self.dropout(ent_embeds[right_entity])
        real_rel_embeds = right_ent_embeds - left_ent_embeds

        # [few/batch, max_neighbor, k, embed_dim]
        real_rel_embeds = real_rel_embeds.unsqueeze(1).repeat(1, relations_out.size(1), 1, 1) 
        #real_rel_embeds = self.ent_embeddings.weight.data[left_entity] - self.ent_embeddings.weight.data[right_entity]
        #real_rel_embeds = real_rel_embeds.unsqueeze(1).unsqueeze(2).repeat(1, 25, 3, 1)

        # [few/batch, max_neighbor, k, 2 * embed_dim]
        #concat_in = torch.cat((rel_embeds_in, ent_embeds_in), dim=-1)
        #concat_out = torch.cat((rel_embeds_out, ent_embeds_out), dim=-1)
        # [few/batch, max_neighbor, k, 1]
        #tmp_in = self.fc1(concat_in)
        #tmp_out = self.fc2(concat_out)

        # concat的方式 [few/batch, max_neighbor, k, 1]
        #atten_in = torch.softmax(tmp_in, dim=1)
        #atten_out = torch.softmax(tmp_out, dim=1)
        #atten_in = torch.cosine_similarity(rel_embeds_in, real_rel_embeds, dim=-1).unsqueeze(3)
        #atten_out = torch.cosine_similarity(rel_embeds_out, real_rel_embeds, dim=-1).unsqueeze(3)
        #score_in = self.Bilinear(rel_embeds_in, real_rel_embeds)
        #atten_in = torch.softmax(score_in, dim=1)
        
        concat_in = torch.cat((rel_embeds_in, real_rel_embeds), dim=-1)
        tmp_in = self.attention1(concat_in) 
        atten_in1 = torch.softmax(tmp_in, dim=1)
        #atten_in2 = torch.softmax(tmp_in, dim=2)
 
        concat_out = torch.cat((rel_embeds_out, real_rel_embeds), dim=-1)
        tmp_out = self.attention2(concat_out)
        atten_out1 = torch.softmax(tmp_out,dim=1)
        #atten_out2 = torch.softmax(tmp_out,dim=2)
 
        #score_out = self.Bilinear(rel_embeds_out, real_rel_embeds)
        #atten_out = torch.softmax(score_out, dim=1)
        
        #atten_in = torch.softmax(self.fc1(torch.mul(real_rel_embeds, rel_embeds_in)), dim=1) 
        #atten_out = torch.softmax(self.fc2(torch.mul(real_rel_embeds, rel_embeds_out)), dim=1) 

        # [few/batch, k, 100]
        h_in = torch.sum(atten_in1 * ent_embeds_in, dim=1)
        h_out = torch.sum(atten_out1 * ent_embeds_out, dim=1)
        nu = torch.exp(h_in) + torch.exp(h_out)
        out = (h_in * torch.exp(h_in) + h_out * torch.exp(h_out)) / nu
        return out.tanh()


    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)  batch_size=128
        support: (few, 2)  few=1
        query_left_connections: (batch_size, max_neighbor, 2)
        support_left_connections: (few, max_neighbor, 2)

        return: (batch_size, )
        '''
        # query_left_connections是一个元组，包含出度和入度  ([128, 50, 2], [128, 50, 2])
        ent1 = F.normalize(self.linear1(self.ent_embeddings.weight.data),dim=1).unsqueeze(1)
        ent2 = F.normalize(self.linear2(self.ent_embeddings.weight.data),dim=1).unsqueeze(1)
        ent3 = F.normalize(self.linear3(self.ent_embeddings.weight.data),dim=1).unsqueeze(1)

        rel1 = F.normalize(self.linear1(self.rel_embeddings.weight.data),dim=1).unsqueeze(1)
        rel2 = F.normalize(self.linear2(self.rel_embeddings.weight.data),dim=1).unsqueeze(1)
        rel3 = F.normalize(self.linear3(self.rel_embeddings.weight.data),dim=1).unsqueeze(1)
        #rel_embeds = self.rel_embeddings.weight.unsqueeze(1).expand(-1, self.k, self.embed_dim)
        ent_embeds = torch.cat([ent1, ent2, ent3], dim=1)
        rel_embeds = torch.cat([rel1, rel2, rel3], dim=1)
        # support_left_connection也是一个元组，包含出度和入度  ([1, 50, 2], [1, 50, 2])
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        # 1. 获得实体的表示  [batch_size, embed_dim]
        # query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        # query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        # 关注出度和入度 [batch, k, embed_dim]
        query_left = self.neighbor_encoder(query[:, 0], query[:, 1], query_left_connections, query_left_degrees, rel_embeds, ent_embeds)
        query_right = self.neighbor_encoder(query[:, 0], query[:, 1], query_right_connections, query_right_degrees, rel_embeds, ent_embeds)

        # [few, embed_dim]
        # support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        # support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        # 关注出度和入度 [few, k, embed_dim]
        support_left = self.neighbor_encoder(support[:, 0], support[:, 1], support_left_connections,
                                             support_left_degrees, rel_embeds, ent_embeds)
        support_right = self.neighbor_encoder(support[:, 0], support[:, 1], support_right_connections,
                                              support_right_degrees, rel_embeds, ent_embeds)

        # 2. 获得实体对的表示 [batch/few, 2*embed_dim]
        # query_neighbor = torch.cat((query_left, query_right), dim=-1)
        # support_neighbor = torch.cat((support_left, support_right), dim=-1)

        # [batch / few, k, 2 * embed_dim]
        query_neighbor = torch.cat((query_left, query_right), dim=-1)
        support_neighbor = torch.cat((support_left, support_right), dim=-1)
        support = support_neighbor
        query = query_neighbor

        # 3. 对实体对进行编码 [batch/few, 2*embed_dim]
        # support_g = self.support_encoder(support)
        # query_g = self.support_encoder(query)

        # [batch / few, k, 2 * embed_dim]
        support_g = self.support_encoder(support)
        query_g = self.support_encoder(query)

        # 这个平均操作针对few-shot的情况
        support_g = torch.mean(support_g, dim=0, keepdim=True)
        # support_g = support
        # query_g = query

        # 解耦后的embedding
        #support_g = support_g.view(support_g.size(0), -1)
        #query_g = query_g.view(query_g.size(0), -1)
        # 4. 对query中的实体对进行编码 [batch_size, 2*embed_dim]
        # query_f = self.query_encoder(support_g, query_g)
        query_f1 = self.query_encoder(support_g.squeeze(0), query_g[:,0])
        query_f2 = self.query_encoder(support_g.squeeze(0), query_g[:,1])
        query_f3 = self.query_encoder(support_g.squeeze(0), query_g[:,2])
      
        # 5. 计算相似度 [batch]
        matching_scores1 = torch.matmul(query_f1, support_g[:,0].t()).squeeze()
        matching_scores2 = torch.matmul(query_f2, support_g[:,1].t()).squeeze()
        matching_scores3 = torch.matmul(query_f3, support_g[:,2].t()).squeeze()
        return torch.max(torch.max(matching_scores1, matching_scores2), matching_scores3)
