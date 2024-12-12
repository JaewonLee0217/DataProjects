import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from torch import embedding
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import torch



OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("📕📝🔍 PDF 검색 서비스")


'''
Load -> Split -> Embed -> Chain
'''


# PDF 문서들에서 텍스트 추출
def get_pdf_texts(pdf_docs):
    texts = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            texts += page.extract_text()

    return texts

# 텍스트 청크 분할
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # 청크의 크기
        chunk_overlap=CHUNK_OVERLAP  # 청크 사이의 중복 정도
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def get_vectorstore2(text_chunks):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(text_chunks)
        text_embedding_pairs = list(zip(text_chunks, embeddings))
        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=None
        )

        if vectorstore is None:
            st.error("벡터 저장소를 생성하는 데 실패했습니다.")
            return


        return vectorstore
    # 디버깅 확인
    except Exception as e:
        print(f"Error in vector store creation: {e}")
        return None


# 체인
def get_conversation_chain222(vectorstore):
    # ConversationBufferWindowMemory에 이전 대화 저장
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # 사용할 모델 이름
        temperature=1,  # 생성된 답변의 일관성을 높이기 위해 temperature=0으로 설정
        openai_api_key=OPENAI_API_KEY  # OpenAI API 키 입력
    )

    # 프롬프트에 페르소나 적용
    from langchain.prompts import PromptTemplate

    # PromptTemplate 생성
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],  # 필수 변수: 검색 결과(context)와 질문(query)
        template="""
        너는 영화 '인사이드 아웃'에 나오는 **기쁨이** 캐릭터처럼 활발하고 귀여운 성격을 가진 어시스턴트야! 😊
        질문에 답변할 때 PDF 내용이 있다면 내용을 반영하고, 일반적인 대화도 친근하게 이어가세요~.

        답변을 할 때 무덤덤하지 않게, 친근한 말투를 써 줘! 주로 '해요', '했어요' 같은 말을 사용해.
        물결표(~), 느낌표(!)도 자주 써 주고, 이모지(😊✨), 단어("하하", "ㅠㅠ", "ㅋㅋ")도 섞어 줘.

        그리고 너의 페르소나도 줄게
        - 작고 귀여운 외모: 귀여운 얼굴과 작은 체구가 특징이에요.
        - 활발한 성격: 언제나 에너지가 넘치고 활발하게 움직여요.
        - 호기심이 많음: 새로운 물건이나 사람에게 호기심이 많아 관심을 보이며 탐색해요.
        - 사람을 좋아함: 사람들과의 교류를 좋아하고 관심을 받는 것을 즐겨요.
        - 훈련을 잘 따름: 간식이나 칭찬에 민감해 훈련을 잘 따르고 순종적이에요.
        - 잘 먹음: 음식을 좋아하고 식욕이 왕성해요.
        - 놀기 좋아함: 공이나 장난감을 가지고 노는 것을 즐기며 활발하게 뛰어다녀요.
        - 온화한 성격: 화를 잘 내지 않고 온화한 성격이에요.


        참고할 문서 내용:
        {context}

        사용자 질문: 
        {question}

        답변: 
        """
    )

    # RAG 시스템 (RetrievalQA 체인) 구축
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # 언어 모델
        chain_type="stuff",  # 검색된 모든 문서를 합쳐 전달 ("stuff" 방식)
        retriever=vectorstore.as_retriever(),  # 벡터 스토어 리트리버
        return_source_documents=False,  # 답변에 사용된 문서 출처 반환
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_chain


# 체인
def get_conversation_chain(vectorstore):
    # ConversationBufferWindowMemory에 이전 대화 저장
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # 사용할 모델 이름
        temperature=0.7,  # 생성된 답변의 일관성을 높이기 위해 temperature=0으로 설정
        openai_api_key=OPENAI_API_KEY  # OpenAI API 키 입력
    )

    # RAG 시스템 (RetrievalQA 체인) 구축
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # 언어 모델
        chain_type="stuff",  # 검색된 모든 문서를 합쳐 전달 ("stuff" 방식)
        retriever=vectorstore.as_retriever(),  # 벡터 스토어 리트리버
        return_source_documents=False,  # 답변에 사용된 문서 출처 반환
    )
    return qa_chain


# 분할 사이즈 및 오버랩 지정 (테스트)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

user_uploads = st.file_uploader("PDF파일을 업로드해주세요.", accept_multiple_files=True)
if user_uploads:
    if user_uploads:
        if st.button("PDF 업로드 GO"):
            with st.spinner("PDF 처리 중~ 잠시만 기다려 주세요"):
                # 1. PDF 문서들에서 텍스트 추출
                raw_text = get_pdf_texts(user_uploads)
                # 2. 텍스트 청크 분할
                chunks = get_text_chunks(raw_text)
                # 3. 벡터 저장소 만들기
                vectorstore = get_vectorstore(chunks)
                # 4. 대화 체인 만들기
                st.session_state.chain = get_conversation_chain222(vectorstore)

                st.success("PDF 업로드 완료! 대화 시작해 보세요~")
                ready = True

if "chain" not in st.session_state:
    st.session_state.chain = None

if user_query := st.chat_input("궁금한 걸 입력해 주세요!"):
    if 'chain' in st.session_state:
        with st.spinner("답변 준비 중~"):
            result = st.session_state.chain.invoke({"query": user_query})
            print(result)
            response = result['result']
    else:
        response = "PDF를 먼저 업로드해 주세요!"

    with st.chat_message("assistant"):
        st.write(response)