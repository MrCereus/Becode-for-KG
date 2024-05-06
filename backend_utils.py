import numpy as np
import dgl
import torch
from bidict import bidict
import os
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import py2neo
import re
import jieba 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class KG:
    def __init__(self):
        self.rel_ids = bidict()
        self.ent_ids = bidict()
        self.er_e = dict()
        self.ee_r = dict()
        self.edges = set()
        self.rels = set()

    def construct_graph(self):
        ent_graph = dgl.graph(list(self.edges))
        self.ent_graph = dgl.to_bidirected(ent_graph).to_simple()


class EAData:
    def __init__(self, loc, bi=True):
        self.kg = [KG(), KG()]
        self.seed_pair = bidict()
        self.test_pair = bidict()
        self.loc = loc
        self.attrs = {}
        self.load_dbp(bi)
        self.test_pair.update(self.seed_pair)
        self.seed_pair = bidict()
        stop_words = []
        with open(loc+'stopwords_full.txt', encoding='utf-8') as f: # 可根据需要打开停用词库，然后加上不想显示的词语
            for line in f.readlines():
                stop_words.append(line.replace("\n",""))
        stop_words += list(nltk.corpus.stopwords.words('english'))
        self.stop_words = set(stop_words)
        

    def load_dbp(self, bi):
        for i in range(2):
            with open(self.loc+'rel_ids_{}'.format(i+1), encoding='UTF-8') as f:
                for line in f.readlines():
                    ids, rel = line.strip().split('\t')
                    self.kg[i].rel_ids[int(ids)] = rel.split('property/')[-1]

            with open(self.loc+'ent_ids_{}'.format(i+1), encoding='UTF-8') as f:
                for line in f.readlines():
                    ids, ent = line.strip().split('\t')
                    self.kg[i].ent_ids[int(ids)] = ent.split('resource/')[-1].replace('_', ' ')

            with open(self.loc+'triples_{}'.format(i+1), encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')
                    head, rel, tail = int(head), int(rel), int(tail)
                    # head, tail = self.kg[i].ent_ids.inv[head], self.kg[i].ent_ids.inv[tail]
                    self.kg[i].edges.add((head, tail))
                    self.kg[i].rels.add((head, rel, tail))
                    self.kg[i].er_e[(head, rel)] = tail
                    if (head, tail) not in self.kg[i].ee_r.keys():
                        self.kg[i].ee_r[(head, tail)] = [rel]
                    else:
                        self.kg[i].ee_r[(head, tail)].append(rel)
                    if bi:
                        self.kg[i].er_e[(tail, rel)] = head
                self.kg[i].construct_graph()
            with open(self.loc+f'attr_triples_{str(i+1)}', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')[:3]
                    ent = self.kg[i].ent_ids.inv[head.split('resource/')[-1].replace('_', ' ')]
                    prop = rel.split('property/')[-1]
                    match = re.search(r'"(.*?)"', tail)
                    if match:
                        val = match.group(1)
                    else:
                        val = tail
                    if ent not in self.attrs.keys():
                        self.attrs[ent] = val
                    else:
                        self.attrs[ent] += " " + val
                k = self.attrs.keys()
            with open(self.loc+f'atts_properties_{str(i+1)}.txt', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')[:3]
                    ent = self.kg[i].ent_ids.inv[head.split('resource/')[-1].replace('_', ' ')]
                    prop = rel.split('property/')[-1]
                    match = re.search(r'"(.*?)"', tail)
                    if match:
                        val = match.group(1)
                    else:
                        val = tail
                    if ent not in self.attrs.keys():
                        self.attrs[ent] = val
                    elif ent not in k:
                        self.attrs[ent] += " " + val

        with open(self.loc+'sup_pairs', 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                head, tail = line.strip().split('\t')
                self.seed_pair[int(head)] = int(tail)

        with open(self.loc+'new_ref_pairs', 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                head, tail = line.strip().split('\t')
                self.test_pair[int(head)] = int(tail)
        if os.path.exists(self.loc+'hard_pairs.txt'):
            self.hard_pair = {}
            with open(self.loc+'hard_pairs.txt', 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, tail = line.strip().split('\t')
                    self.hard_pair[int(head)] = int(tail)

class GWEA():
    def __init__(self, data, use_attr=True, use_trans=False, hard_pair=False):
        self.iters = 0
        self.data = data
        self.candi = self.data.test_pair.copy()
        self.graph1 = self.data.kg[0].ent_graph
        self.graph2 = self.data.kg[1].ent_graph
        self.rel_list = [list(self.data.kg[0].rel_ids),list(self.data.kg[1].rel_ids)]
        self.ent_ids1 = bidict()
        self.ent_ids2 = bidict()
        self.rel_emb = torch.tensor(np.load(self.data.loc+'rel_emb.npy'))
        self.ent_emb = torch.tensor(np.load(self.data.loc+'ent_emb.npy')).float()
        self.attr_emb = torch.tensor(np.load(self.data.loc+'attr_emb.npy')).float()
        self.ent_emb = F.normalize(self.ent_emb, p=2, dim=1)
        self.attr_emb = F.normalize(self.attr_emb, p=2, dim=1)
        self.rel_emb = F.normalize(self.rel_emb, p=2, dim=1)
        rand_ind = np.random.permutation(len(data.test_pair))
        self.test_pair = {}
        for i, ind in enumerate(rand_ind):
            self.test_pair[i] = ind
        for i, (ent1, ent2) in enumerate(self.candi.items()):
            self.ent_ids1[i] = ent1
            self.ent_ids2[i] = ent2
        if hard_pair:
            self.hard_pair = {}
            for k, v in data.hard_pair.items():
                self.hard_pair[self.ent_ids1.inv[k]] = self.ent_ids2.inv[v]
        self.n = len(self.ent_ids1)
        self.ent_ids2 = bidict(sorted(self.ent_ids2.items()))
        self.cost_st_feat = (1 + self.ent_emb[list(self.ent_ids1.values())]@self.ent_emb[list(self.ent_ids2.values())].T) / 2
        self.cost_st_attr = (1+self.attr_emb[list(self.ent_ids1.values())]@self.attr_emb[list(self.ent_ids2.values())].T)/2

class Algo:
    def __init__(self, gwea:GWEA):
        self.gwea = gwea
        self.w = [0.3,0.4,0.3,0.05]
        self.final_res_dict = {}
        self.round = 0
        url = "http://127.0.0.1:7474"
        username = "neo4j"
        password = "kgdemo"
        self.graph = py2neo.Graph(url, auth=(username, password))
    
    def get_nei_rel(self, rels):
        rel_emb = torch.tensor(self.gwea.rel_emb)
        rel_embs = torch.tensor([])
        # rel = set([r for i in range(len(rels)) for r in rels[i]])
        for r in rels:
            rel_embs = torch.cat((rel_embs, rel_emb[[r]]), dim=0)
        return rel_embs

    def maximum_matching_average_similarity(self, embeddings1, embeddings2):
        similarity_matrix = 1 - cdist(embeddings1, embeddings2, metric='cosine')
        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        total_similarity = similarity_matrix[row_ind, col_ind].sum()
        average_similarity = total_similarity / len(row_ind)
        return row_ind, col_ind, average_similarity, similarity_matrix

    def cal_outside_neighbor(self, graph, neighbors):
        neighbors_np = np.array(neighbors)
        res = []
        for nei in neighbors:
            total_nei = torch.cat((graph.predecessors(nei), graph.successors(nei)), dim=0).unique().numpy()
            count = np.sum(np.isin(total_nei,neighbors_np,invert=True))
            res.append(count)
        return res

    def cal_stru_sim(self,id1,id2):
        e1n = torch.cat((self.gwea.graph1.predecessors(id1), self.gwea.graph1.successors(id1)), dim=0).unique().tolist()
        e2n = torch.cat((self.gwea.graph2.predecessors(id2), self.gwea.graph2.successors(id2)), dim=0).unique().tolist()
        e1_emb = self.gwea.ent_emb[e1n] # 获得一跳邻居
        e2_emb = self.gwea.ent_emb[e2n]
        e1_emb = np.array(e1_emb)
        e2_emb = np.array(e2_emb)
        row_ind, col_ind, stru_sim, similarity_matrix = self.maximum_matching_average_similarity(e1_emb,e2_emb)
        return row_ind, col_ind, stru_sim, similarity_matrix, e1n, e2n

    def cal_rel_sim(self,id1,id2):
        rels_id1 = []
        for i in self.gwea.graph1.successors(id1):
            if (id1, int(i)) in self.gwea.data.kg[0].ee_r.keys():
                rels_id1.append(self.gwea.data.kg[0].ee_r[(id1, int(i))])
            if (int(i), id1) in self.gwea.data.kg[0].ee_r.keys():
                rels_id1.append(self.gwea.data.kg[0].ee_r[(int(i), id1)])
        rels_id1 = list(set([r for i in range(len(rels_id1)) for r in rels_id1[i]]))
        rels_id2 = []
        for i in self.gwea.graph2.successors(id2):
            if (id2, int(i)) in self.gwea.data.kg[1].ee_r.keys():
                rels_id2.append(self.gwea.data.kg[1].ee_r[(id2, int(i))])
            if (int(i), id2) in self.gwea.data.kg[1].ee_r.keys():
                rels_id2.append(self.gwea.data.kg[1].ee_r[(int(i), id2)])
        rels_id2 = list(set([r for i in range(len(rels_id2)) for r in rels_id2[i]]))
        a = self.get_nei_rel(rels_id1) # 获得一跳邻居
        b = self.get_nei_rel(rels_id2)
        a = np.array(a)
        b = np.array(b)
        if a.shape[0] == 0 or b.shape[0] == 0:
            return 1/1000
        row_ind, col_ind, rel_sim, similarity_matrix = self.maximum_matching_average_similarity(a,b)
        return row_ind, col_ind, rel_sim, similarity_matrix, rels_id1, rels_id2

    def cal_name_sim(self,id1,id2):
        res = (1 + self.gwea.ent_emb[id1]@self.gwea.ent_emb[id2].T) / 2
        return float(res)

    def cal_attr_sim(self,id1,id2):
        res = (1 + self.gwea.attr_emb[id1]@self.gwea.attr_emb[id2].T) / 2
        return float(res)
    
    def cal_mix_sim(self, id1, id2):
        sim_name = self.cal_name_sim(id1, id2)
        sim_attr = self.cal_attr_sim(id1, id2)
        sim_stru = self.cal_stru_sim(id1, id2)[2]
        sim_rel = self.cal_rel_sim(id1, id2)[2]
        sim_mix = self.w[0]*sim_name+self.w[1]*sim_attr+self.w[2]*sim_stru+self.w[3]*sim_rel
        return sim_mix
    
    def cal_word_cloud(self, id1):
        docs = []
        words = jieba.lcut(self.gwea.data.attrs[id1], cut_all=False)
        words = [w for w in words if w not in self.gwea.data.stop_words]
        docs.append(" ".join(words))
        for v in self.final_res_dict[id1]:
            words = jieba.lcut(self.gwea.data.attrs[v], cut_all=False)
            words = [w for w in words if w not in self.gwea.data.stop_words]
            docs.append(" ".join(words))
        vectorizer = TfidfVectorizer(stop_words=self.gwea.data.stop_words, use_idf= True)
        X = vectorizer.fit_transform(docs) # 计算词频矩阵
        attr_words = vectorizer.get_feature_names_out() # 查看词汇表
        attr_freqs = X.toarray() # 查看词频矩阵
        attr_res = {}
        attr_res[id1] = []
        for i in range(len(attr_words)):
            if attr_freqs[0][i] != 0:
                attr_res[id1].append({
                    "x": attr_words[i],
                    "value": attr_freqs[0][i] * 10,
                    "category": "1"
                })
        for i in range(1,6):
            id2 = self.final_res_dict[id1][i - 1]
            attr_res[id2] = []
            for j in range(len(attr_words)):
                if attr_freqs[i][j] != 0:
                    attr_res[id2].append({
                        "x": attr_words[j],
                        "value": attr_freqs[i][j] * 10,
                        "category": "1"
                    })
        return attr_res

    def cal_top_5_res(self):
        arr = self.gwea.cost_st_attr+self.gwea.cost_st_feat
        top_100_indices = np.argsort(arr, axis=1)[:, -100:]
        start = self.round * 100
        test_num = 100
        ent1 = [self.gwea.ent_ids1[i] for i in range(start, start + test_num)]
        ent2 = [[self.gwea.ent_ids2[int(i)] for i in top_100_indices[int(j)]] for j in range(test_num)]
        sim_feat = []
        sim_attr = []
        sim_stru = []
        sim_rel= []
        for i in range(len(ent1)):
            tmp_feat = []
            tmp_attr = []
            for j in top_100_indices[i]:
                tmp_feat.append(self.gwea.cost_st_feat[int(i)][int(j)])
                tmp_attr.append(self.gwea.cost_st_attr[int(i)][int(j)])
            sum_tmp_feat = np.sum(tmp_feat)
            tmp_feat = np.array(tmp_feat) / sum_tmp_feat
            sum_tmp_attr = np.sum(tmp_attr)
            tmp_attr = np.array(tmp_attr) / sum_tmp_attr
            sim_feat.append(tmp_feat)
            sim_attr.append(tmp_attr)
        sim_feat = np.array(sim_feat)
        sim_attr = np.array(sim_attr)
        for i in range(len(ent1)):
            id1 = ent1[i]
            tmp = []
            for id2 in ent2[i]:
                tmp.append(self.cal_stru_sim(id1,id2)[2])
            sum_tmp = np.sum(tmp)
            tmp = np.array(tmp) / sum_tmp
            sim_stru.append(tmp)
        sim_stru = np.array(sim_stru)

        for i in range(len(ent1)):
            id1 = ent1[i]
            tmp = []
            for id2 in ent2[i]:
                tmp.append(self.cal_rel_sim(id1,id2)[2])
            sum_tmp = np.sum(tmp)
            tmp = np.array(tmp) / sum_tmp
            sim_rel.append(tmp)
        sim_rel = np.array(sim_rel)
        final_res = self.w[0]*sim_feat+self.w[1]*sim_attr+self.w[2]*sim_stru+self.w[3]*sim_rel
        top_5_res_indices = np.argsort(final_res, axis=1)[:, -5:]
        ent2_np = np.take_along_axis(np.array(ent2),top_5_res_indices,axis=1)
        final_res_dict = {}
        for i in range(len(ent1)):
            final_res_dict[int(ent1[i])] = ent2_np[i][::-1].tolist()
        self.final_res_dict = final_res_dict
        return final_res_dict
    
    def get_table_data(self, round = 0):
        self.round = round
        final_res_dict = self.cal_top_5_res()
        table_data = []
        for key, value in final_res_dict.items():
            id1 = int(key)
            id2 = int(value[0])
            sim_mix = self.cal_mix_sim(id1, id2)
            table_data.append({
                "KG1": self.gwea.data.kg[0].ent_ids[id1],
                "ID1": id1,
                "KG2": self.gwea.data.kg[1].ent_ids[id2],
                "ID2": id2,
                "Sim": sim_mix
            })
        return table_data

    def get_sim_data(self, id1):
        id1 = int(id1)
        mix = []
        name = []
        attr = []
        stru = []
        rel = []
        word_clouds = self.cal_word_cloud(id1)
        for id2 in self.final_res_dict[id1]:
            id2 = int(id2)
            # name
            sim_name = self.cal_name_sim(id1, id2)
            name.append({
                "ID1": id1,
                "ID2": id2,
                "Sim": sim_name,
                "Res": [self.gwea.data.kg[0].ent_ids[id1],self.gwea.data.kg[1].ent_ids[id2]]
            })

            # attr
            sim_attr = self.cal_attr_sim(id1, id2)
            attr.append({
                "ID1": id1,
                "ID2": id2,
                "Sim": sim_attr,
                "Res": [word_clouds[id1],word_clouds[id2]]
            })

            # stru
            row_ind, col_ind, sim_stru, similarity_matrix, e1n, e2n = self.cal_stru_sim(id1, id2)
            tmp_res = [[],[]]
            align_pair = []
            for row, col in zip(row_ind, col_ind):
                tmp_res[0].append({
                    "KG1": self.gwea.data.kg[0].ent_ids[int(e1n[row])],
                    "Sim": similarity_matrix[row][col]
                })
                tmp_res[1].append({
                    "KG2": self.gwea.data.kg[1].ent_ids[int(e2n[col])],
                    "Sim": similarity_matrix[row][col]
                })
                align_pair.append([int(e1n[row]),int(e2n[col])])
            stru.append({
                "ID1": id1,
                "ID2": id2,
                "centerNodePair":[id1, id2],
                "alignNodePair":align_pair,
                "Sim": sim_stru,
                "Res": tmp_res
            })
            # rel
            row_ind, col_ind, sim_rel, similarity_matrix, rel1, rel2 = self.cal_rel_sim(id1, id2)
            tmp_res = [[],[]]
            for row, col in zip(row_ind, col_ind):
                tmp_res[0].append({
                    "KG1": self.gwea.data.kg[0].rel_ids[int(rel1[row])],
                    "Sim": similarity_matrix[row][col]
                })
                tmp_res[1].append({
                    "KG2": self.gwea.data.kg[1].rel_ids[int(rel2[col])],
                    "Sim": similarity_matrix[row][col]
                })
            rel.append({
                "ID1": id1,
                "ID2": id2,
                "Sim": sim_rel,
                "Res": tmp_res
            })

            # mix
            mix.append({
                "ID1": id1,
                "ID2": id2,
                "Sim": self.w[0]*sim_name+self.w[1]*sim_attr+self.w[2]*sim_stru+self.w[3]*sim_rel,
            })

        res = {
            "sim_mix": mix,
            "name": name,
            "attr": attr,
            "stru": stru,
            "rel": rel
        }
        return res
    def get_force_graph_data(self, id1, id2):
        def get_subgraph(id):
                cypher = f'''
                    MATCH (n) where n.id={id} \
                    CALL apoc.path.subgraphAll(n, {{\
                        maxLevel: 1\
                    }}) \
                    YIELD relationships \
                    unwind relationships as r_ WITH DISTINCT r_ \
                    return type(r_) as r_name, id(r_) as r_id, \
                    startNode(r_).id as x_id, startNode(r_).name as x_name, \
                    endNode(r_).id as y_id, endNode(r_).name as y_name, \
                    labels(startNode(r_)) as x_labels, labels(endNode(r_)) as y_labels
                '''
                df = self.graph.run(cypher).to_data_frame()
                rel_list = df.values.tolist()
                nodes = {}
                edges = {}
                for rel in rel_list:
                    if rel[2] not in nodes.keys():
                        nodes[rel[2]] = rel[3]
                    if rel[4] not in nodes.keys():
                        nodes[rel[4]] = rel[5]
                    if tuple([rel[2], rel[4]]) not in edges.keys():
                        edges[tuple([rel[2], rel[4]])] = [rel[0]]
                    else:
                        edges[tuple([rel[2], rel[4]])].append(rel[0])
                res = {
                    "nodes":[],
                    "edges":[]
                }
                for key, value in nodes.items():
                    res["nodes"].append({
                        "id" : key,
                        "name": value
                    })
                for key, value in edges.items():
                    res["edges"].append({
                        "source" : key[0],
                        "target": key[1],
                        "rels" : value
                    })
                return res
        return [get_subgraph(id1),get_subgraph(id2)]
