'''计算每一个人数据库已存向量所属的文件名列表，用于展示选择框'''
from pinecone import Pinecone


def get_doc_names(index_name):
    pc = Pinecone()
    index1 = pc.Index(index_name)

    # 定义一个集合来存储独立的doc_name
    unique_doc_names = set()

    # 获取索引统计信息
    index_stats = index1.describe_index_stats()
    print(index_stats)
    # 获取向量总数
    try:
        total_vectors = index_stats['namespaces']['']['vector_count']
    except KeyError:
        return unique_doc_names


    # 查询所有向量
    query_response = index1.query(
        top_k=total_vectors,
        include_metadata=True,
        vector=list(range(index1.describe_index_stats()['dimension'])),
    )

    print(len(query_response['matches']))
    # 遍历查询结果并提取metadata中的doc_name
    for result in query_response['matches']:
        doc_name = result['metadata'].get('source', None)
        if doc_name:
            unique_doc_names.add(doc_name)

    # 输出所有独立的doc_name
    return unique_doc_names




