import re
import dotenv
import cohere
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()


pc = Pinecone()
embed_model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=embed_model, openai_api_base="https://api.gptsapi.net/v1")

# 这是重排序的工具 Rerank3 用于对文档进行重新排序
co = cohere.Client()
# 异步不异步的都定义一下，如果选择是单文档逐个分析，就启动异步，如果是多文档一起分析，就不启动异步

def normalize_filename(filename):
    # 使用正则表达式匹配文件名中的最后一部分
    match = re.search(r'[^\\/"<>|\r\n]+\.pdf$', filename)
    if match:
        return match.group()
    else:
        return filename


def reformat_list(items):
    result = []
    for index, item in enumerate(items, 1):  # 从1开始编号
        prefix_removed = item.split('_', 1)[1]  # 去除数字和下划线
        formatted_item = f"{index}. {prefix_removed}"  # 添加编号
        result.append(formatted_item)
    return "\n\n".join(result)  # 将列表中的字符串合并为一个大字符串


# 定义一个函数，用于将带有双反斜杠的字符串转换为原始字符串
def unescape_string(s):
    return re.sub(r'\\\\', r'\\', s)
# 处理函数

# 用于异步获取知识库的函数，输入是用户的问题，输出是经过重排序的知识库中的文档
def get_docs(query: str, top_k: int, index_name, rerank_top_k, filter_metadata=None):
    index = pc.Index(index_name)
    # # raw_paths = [r"{path}".format(path=p) for p in filter_metadata]
    # # print(raw_paths)
    # # 对list2中的字符串进行转换
    # unescaped_list2 = [unescape_string(s) for s in filter_metadata]

    # 对查询进行编码
    xq = embeddings.embed_query(query)
    # 在 Pinecone 索引中搜索
    res = index.query(vector=xq, top_k=top_k, include_metadata=True, filter={"source": {"$in":filter_metadata}})
    # 在 Pinecone 索引中搜索
    # res = index_name.query(vector=xq, top_k=top_k, include_metadata=True)

    # 检查是否有匹配的文档
    if len(res["matches"]) == 0:
        return ["1_无相关资讯内容"]

    # 提取文档文本
    docs = {f"{i}_{x['metadata']['text']}": i for i, x in enumerate(res["matches"])}
    sources = [f"{x['metadata']['source']}" for i, x in enumerate(res["matches"])]
    normalized_sources = [normalize_filename(filename) for filename in sources]

    # 打印处理后的文件名列表
    # print(normalized_sources)


    pages = [f"{x['metadata']['page']}" for i, x in enumerate(res["matches"])]
    reverse_docs = {v: k for k, v in docs.items()}



    # 对文档进行重新排序
    rerank_docs = co.rerank(
        query=query, documents=list(docs.keys()), top_n=rerank_top_k, model="rerank-multilingual-v3.0"
    )
    # rerank_docs = co.rerank(
    #     query=query, documents=list(docs.keys()), top_n=rerank_top_k, model="rerank-multilingual-v3.0"
    # )

    # 获取重新排序后的文档内容
    reranked_index = [doc.index for doc in rerank_docs.results]
    # 使用重新排序后的文档的索引从反向字典中获取对应的文本
    reranked_texts = [reverse_docs[index] for index in reranked_index]
    concatenated_texts = []

    for reranked_text, source, page_num in zip(reranked_texts, normalized_sources, pages):
        concatenated_text = reranked_text + "\n\n" + "【" + source + "】" + "→→→" + "第" + page_num + "页"
        concatenated_texts.append(concatenated_text)

    return concatenated_texts



def return_chunks(user_input, index, top_k, rerank_top_k, filter_source):
    try:
        prompt_with_context = get_docs(query=user_input, top_k=top_k, index_name=index, rerank_top_k=rerank_top_k, filter_metadata=filter_source)
        formatted_string = reformat_list(prompt_with_context)
        origin_txt = "以下是原始文本:" + '\n' + formatted_string
        return origin_txt

    except Exception as e:
        print(e)
        error_message = "文本检索过程出现问题,未返回任何有效信息,请重试"
        return error_message










