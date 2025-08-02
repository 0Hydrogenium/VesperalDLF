import time

import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

from utils.GeneralTool import GeneralTool
from utils.database.CypherDriver import CypherDriver
from utils.graph.GraphTool import GraphTool
from utils.llm_api.LLMAPIClient import LLMAPIClient
from utils.nlp.NLPTool import NLPTool
from utils.visualizer.PublicOpinionVisualizer import PublicOpinionVisualizer

if __name__ == '__main__':

    target_keyword = "特斯拉 事故"
    folder_data_path = "@/data/weibo_automobile_company_public_opinion/main_weibo".replace("@", GeneralTool.root_path)
    company_keyword_item_list = []
    time_item_list = []
    text_len_item_list = []
    # for year in range(2020, 2026):
    #     data_path = f"{folder_data_path}/main_weibo_{year}.csv"
    #     df = pd.read_csv(data_path, encoding='ANSI')
    #     df = df.loc[:, [
    #        "id",
    #        "user_id",
    #        "screen_name",
    #        "text",
    #        "keyword",
    #        "created_at",
    #        "attitudes_count",
    #        "comments_count",
    #        "reposts_count",
    #        "user_authentication",
    #        "vip_type",
    #        "vip_level",
    #        "followers_count",
    #        "friends_count",
    #        "total_like_count",
    #        "total_repost_count",
    #        "total_comment_count"
    #    ]]
    #     keyword_list = [
    #         "特斯拉 事故",
    #         "特斯拉 维权",
    #         "问界 事故",
    #         "问界 维权",
    #         "小米SU7 事故",
    #         "小米SU7 维权",
    #         "小鹏汽车 事故",
    #         "小鹏汽车 维权",
    #         "极氪 事故",
    #         "极氪 维权",
    #         "哪吒汽车 事故",
    #         "哪吒汽车 维权",
    #         "哪吒汽车 破产",
    #         "理想汽车 维权",
    #         "蔚来 事故",
    #         "比亚迪 事故",
    #     ]
    #     df["keyword"] = df["keyword"].replace({
    #         "特斯拉 女车主": "特斯拉 维权",
    #         "特斯拉 刹车失灵": "特斯拉 事故",
    #         "问界 起火": "问界 事故",
    #         "问界 质量问题": "问界 维权",
    #         "问界 品控": "问界 维权",
    #         "小米SU7 爆燃": "小米SU7 事故",
    #         "小米SU7 撞车": "小米SU7 事故",
    #         "小米SU7 品控": "小米SU7 维权",
    #         "比亚迪 起火": "比亚迪 事故",
    #         "比亚迪 自燃": "比亚迪 事故",
    #         "比亚迪 火灾": "比亚迪 事故",
    #         "理想车主 投诉": "理想汽车 维权",
    #         "理想汽车 质量问题": "理想汽车 维权",
    #         "理想ONE 断轴": "理想汽车 维权",
    #         "理想汽车 悬架断裂": "理想汽车 维权",
    #         "小鹏汽车 撞车": "小鹏汽车 事故",
    #         "小鹏汽车 撞护栏": "小鹏汽车 事故",
    #         "小鹏汽车 质量问题": "小鹏汽车 维权",
    #         "蔚来 撞路柱": "蔚来 事故",
    #         "蔚来 自动驾驶 事故": "蔚来 事故",
    #         "蔚来 撞车": "蔚来 事故",
    #         "蔚来 自动驾驶 死亡": "蔚来 事故",
    #         "极氪 车机黑屏": "极氪 维权",
    #         "极氪 质量问题": "极氪 维权",
    #         "极氪 品控": "极氪 维权",
    #         "哪吒汽车 质量问题": "哪吒汽车 维权"
    #     })
    #
    #     df = df.drop_duplicates()
    #
    #     df = df[df["keyword"].isin(keyword_list)].reset_index(drop=True)
    #     company_keyword_item_list += df["keyword"].tolist()
    #
    #     df = df[df["keyword"] == target_keyword].reset_index(drop=True)
    #
    #     df["created_at"] = pd.to_datetime(df["created_at"])
    #     df["created_at"] = df["created_at"].dt.strftime('%Y-%m')
    #     time_item_list += df["created_at"].tolist()
    #
    #     df["text_len"] = df["text"].str.len()
    #     text_len_item_list += df["text_len"].tolist()
    #
    #     for index in tqdm(range(len(df))):
    #
    #         # 过滤无效短文本
    #         min_text_len = 60
    #         if len(df.loc[index, "text"]) < min_text_len:
    #             print(f"{df.loc[index, 'text']} skip")
    #             continue
    #
    #         # 创建博主节点
    #         # 如果已存在则会更新属性值
    #         CypherDriver.execute_write(
    #             """
    #                 MERGE (n:blogger {weibo_id: $user_id})
    #                 ON CREATE SET
    #                     n.gid = $gid,
    #                     n.name = $screen_name,
    #                     n.user_authentication = $user_authentication,
    #                     n.vip_type = $vip_type,
    #                     n.vip_level = $vip_level,
    #                     n.followers_count = $followers_count,
    #                     n.friends_count = $friends_count,
    #                     n.total_like_count = $total_like_count,
    #                     n.total_repost_count = $total_repost_count,
    #                     n.total_comment_count = $total_comment_count
    #                 ON MATCH SET
    #                     n.gid = $gid,
    #                     n.name = $screen_name,
    #                     n.user_authentication = $user_authentication,
    #                     n.vip_type = $vip_type,
    #                     n.vip_level = $vip_level,
    #                     n.followers_count = $followers_count,
    #                     n.friends_count = $friends_count,
    #                     n.total_like_count = $total_like_count,
    #                     n.total_repost_count = $total_repost_count,
    #                     n.total_comment_count = $total_comment_count
    #             """,
    #             {
    #                 "user_id": str(df.loc[index, "user_id"].item()),
    #                 "gid": GeneralTool.get_uuid(),
    #                 "screen_name": df.loc[index, "screen_name"],
    #                 "user_authentication": df.loc[index, "user_authentication"],
    #                 "vip_type": df.loc[index, "vip_type"],
    #                 "vip_level": df.loc[index, "vip_level"].item(),
    #                 "followers_count": df.loc[index, "followers_count"].item(),
    #                 "friends_count": df.loc[index, "friends_count"].item(),
    #                 "total_like_count": df.loc[index, "total_like_count"].item(),
    #                 "total_repost_count": df.loc[index, "total_repost_count"].item(),
    #                 "total_comment_count": df.loc[index, "total_comment_count"].item()
    #             }
    #         )
    #
    #         # 创建博客节点 + 博主-博客边
    #         CypherDriver.execute_write(
    #             """
    #                 MATCH (n1:blogger {weibo_id: $user_id})
    #                 MERGE (n1)-[r:relation]-(n:blog {
    #                     weibo_id: $id,
    #                     gid: $gid,
    #                     weibo_id: $id,
    #                     text: $text,
    #                     text_len: $text_len,
    #                     created_at: $created_at,
    #                     attitudes_count: $attitudes_count,
    #                     comments_count: $comments_count,
    #                     reposts_count: $reposts_count
    #                 })
    #             """,
    #             {
    #                 "user_id": str(df.loc[index, "user_id"].item()),
    #                 "gid": GeneralTool.get_uuid(),
    #                 "id": str(df.loc[index, "id"].item()),
    #                 "text": df.loc[index, "text"],
    #                 "text_len": df.loc[index, "text_len"].item(),
    #                 "created_at": df.loc[index, "created_at"],
    #                 "attitudes_count": df.loc[index, "attitudes_count"].item(),
    #                 "comments_count": df.loc[index, "comments_count"].item(),
    #                 "reposts_count": df.loc[index, "reposts_count"].item()
    #             }
    #         )
    #
    #         sentences = NLPTool.split_sentences(df.loc[index, "text"])
    #         for sentence in sentences:
    #             # 创建句子节点
    #             CypherDriver.execute_write(
    #                 """
    #                     MATCH (n1:blog {weibo_id: $id})
    #                     MERGE (n1)-[r:relation]-(n:text_sentence {
    #                         text: $text,
    #                         gid: $gid,
    #                         score: 0
    #                     })
    #                 """,
    #                 {
    #                     "id": str(df.loc[index, "id"].item()),
    #                     "gid": GeneralTool.get_uuid(),
    #                     "text": sentence
    #                 }
    #             )
    #
    #     break

    print(Counter(company_keyword_item_list).most_common())
    print(Counter(text_len_item_list).most_common())

    time_count_dict = dict(Counter(time_item_list))
    visualizer = PublicOpinionVisualizer()
    # visualizer.time_curve({target_keyword: time_count_dict}, figsize=(8, 6), save_path=f"./time_curve.png")

    sentence_list = CypherDriver.execute_read(
        """
            MATCH (n:text_sentence) RETURN n
        """, {}
    )
    sentence_list = [x["n"] for x in sentence_list]

    batch_size = 10
    batch_sentence_list = [sentence_list[i: i + batch_size] for i in range(0, len(sentence_list), batch_size)]

    # chunk_size = 20
    # chunk_index = 0
    # embedding_list = []
    # for batch_idx, batch in enumerate(tqdm(batch_sentence_list)):
    #     # 生成当前批次的嵌入
    #     embeddings = LLMAPIClient.execute_qwen3_embedding([x["text"] for x in batch])
    #     embedding_list += [{
    #         "gid": batch[j]["gid"],
    #         "embedding": embeddings[j]
    #     } for j in range(len(batch))]
    #
    #     # 检查是否达到分块保存条件
    #     if (batch_idx + 1) % chunk_size == 0:  # 每处理chunk_size个批次
    #         # 保存当前分块到文件
    #         save_path = f"embeddings/embeddings_chunk_{chunk_index}.pkl"
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(embedding_list, f)
    #
    #         chunk_index += 1  # 增加分块索引
    #         embedding_list = []  # 重置字典释放内存
    #         time_series.sleep(2)
    #
    # # 循环结束后保存剩余数据
    # if len(embedding_list) > 0:
    #     save_path = f"embeddings/embeddings_chunk_{chunk_index}.pkl"
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(embedding_list, f)

    # # 创建合并字典
    # combined_embedding_list = []
    # # 获取所有分块文件（按文件名排序）
    # chunk_files = sorted(glob.glob('embeddings/embeddings_chunk_*.pkl'))
    # for file_path in chunk_files:
    #     try:
    #         with open(file_path, 'rb') as f:
    #             chunk_data = pickle.load(f)
    #             combined_embedding_list += chunk_data
    #             print(f"已加载: {file_path}, 新增条目: {len(chunk_data)}")
    #     except Exception as e:
    #         print(f"加载 {file_path} 时出错: {str(e)}")
    #
    # print(f"总共加载条目数: {len(combined_embedding_list)}")

    # top_k = 10
    # for i in tqdm(range(len(combined_embedding_list))):
    #     cos_similarity_list = []
    #     for j in range(i + 1, len(combined_embedding_list)):
    #
    #         cos_similarity = NLPTool.compute_cosine_similarity(
    #             combined_embedding_list[i]["embedding"],
    #             combined_embedding_list[j]["embedding"]
    #         )
    #         cos_similarity_list.append({
    #             "gid": combined_embedding_list[j]["gid"],
    #             "score": cos_similarity
    #         })
    #
    #     cos_similarity_list.sort(key=lambda x: x["score"], reverse=True)
    #     for m in cos_similarity_list[:top_k]:
    #         CypherDriver.execute_write(
    #             """
    #                 MATCH (n1:text_sentence {gid: $gid_1})
    #                 MATCH (n2:text_sentence {gid: $gid_2})
    #                 MERGE (n1)-[r:similarity {score: $score}]-(n2)
    #             """,
    #             {
    #                 "gid_1": combined_embedding_list[i]["gid"],
    #                 "gid_2": m["gid"],
    #                 "score": m["score"]
    #             }
    #         )

    # similarity_relation_list = CypherDriver.execute_read(
    #     """
    #         MATCH (n1:text_sentence)-[r:similarity]-(n2:text_sentence)
    #         RETURN n1, r, n2
    #     """, {}
    # )
    # page_rank_score_dict = GraphTool.compute_page_rank(similarity_relation_list)
    # for gid, score in page_rank_score_dict.items():
    #     CypherDriver.execute_write(
    #         """
    #             MATCH (n:text_sentence {gid: $gid})
    #             SET n.score = $score
    #         """,
    #         {
    #             "gid": gid,
    #             "score": score
    #         }
    #     )

    import graphistry

    NEO4J = {
        'uri': os.getenv("NEO4J_CONNECTOR_URI"),
        'auth': (
            os.getenv("NEO4J_CONNECTOR_AUTH_USER"),
            os.getenv("NEO4J_CONNECTOR_AUTH_PASSWORD")
        )
    }
    graphistry.register(
        api=3,
        username="0Hydrogenium",
        password="llh20021121",
        protocol="https",
        server="hub.graphistry.com",
        bolt=NEO4J
    )
    g = graphistry.cypher(
        """
            MATCH (n1:text_sentence)-[r:similarity]-(n2:text_sentence)
            RETURN n1, r, n2
        """
    )
    url = g.plot(render=False)
    print(f"View your visualization at: {url}")


