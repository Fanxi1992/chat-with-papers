'''对用户上传的新文献进行分解向量化和存储，成功后返回文献名'''
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pypdf
from time import sleep
import time
from langchain_openai import OpenAIEmbeddings
import re



pc = Pinecone()


embed_model = "text-embedding-3-small"
batch_size = 100


def normalize_filename(filename):
    # 使用正则表达式匹配文件名中的最后一部分
    match = re.search(r'[^\\/"<>|\r\n]+\.pdf$', filename)
    if match:
        return match.group()
    else:
        return filename
def get_microsecond_timestamp():
    return int(time.time() * 1000000)

def process_files(file_uploads, index_name):
    index = pc.Index(index_name)
    max_retries = 3
    retry_count = 0
    # # 将上传的文件保存到内存中
    # file_buffer = io.BytesIO(file_uploads.read())
    # # 创建 PyPDFLoader 实例,并传入文件对象
    # loader = PyPDFLoader(file_buffer)
    # pages = loader.load()
    while retry_count < max_retries:
        try:
            loader = PyPDFLoader(file_uploads)
            pages = loader.load()
            break
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print("File is locked, waiting and retrying...")
                time.sleep(2)
            else:
                error = "解析文章并向量化失败，请重启网页后重新上传"
                return error


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    #
    # print(chunks)
    # print(len(chunks))
    # print(chunks[3])
    # print(chunks[0].page_content)

# test
    for i in range(0, len(chunks), batch_size):
        i_end = min(len(chunks), i + batch_size)
        meta_batch = chunks[i:i_end]
        print(f'本批次添加chunk为(从{i}到{i_end - 1}): {meta_batch}')

        # get ids
        ids_batch = [get_microsecond_timestamp() + j for j in range(len(meta_batch))]

        # get texts to encode
        texts = [x.page_content for x in meta_batch]

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_base="https://api.gptsapi.net/v1")

        max_attempts = 3
        attempt_count = 0
        while attempt_count < max_attempts:
            try:
                doc_result = embeddings.embed_documents(texts)
                break  # 如果成功,跳出循环
            except:
                attempt_count += 1
                if attempt_count < max_attempts:
                    print(f"Embedding failure, waiting 5 seconds... (Attempt {attempt_count}/{max_attempts})")
                    sleep(2)
                else:
                    error = "解析文章并向量化失败,请重启网页后重新上传"
                    return error

        ids_batch_str = [str(num) for num in ids_batch]

        meta_data = [{
            'source': normalize_filename(x.metadata['source']),
            'text': x.page_content,
            'page': x.metadata['page'],
        } for x in meta_batch]

        this_file_name = meta_data[0]['source']

        to_upsert = list(zip(ids_batch_str, doc_result, meta_data))

        # upsert to Pinecone
        index.upsert(vectors=to_upsert)
        print(f'完成文献{this_file_name}的一部分向量化存储')
        time.sleep(0.05)

    return this_file_name



