import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# OpenAI API 키 설정
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# GPT-4 모델 초기화
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# CLIP 모델 초기화
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# BLIP 모델 초기화
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Streamlit 제목
st.title("🖼️ 이미지 선택 및 분석 서비스")

# 이미지 업로드 섹션
uploaded_files = st.file_uploader("두 개의 이미지를 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    # 이미지 로드 및 표시
    images = [Image.open(file) for file in uploaded_files]
    st.image(images, caption=["이미지 1", "이미지 2"], use_column_width=True)

    # 사용자 프롬프트 입력
    user_prompt = st.text_input("프롬프트를 입력하세요 (예: '자연 풍경이 포함된 이미지')")

    if user_prompt and st.button("분석 시작"):
        with st.spinner("이미지 분석 중..."):
            # CLIP 모델로 이미지와 텍스트 유사도 계산
            inputs = clip_processor(text=[user_prompt], images=images, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).tolist()

            # BLIP 모델로 이미지 캡셔닝
            captions = []
            for image in images:
                inputs = blip_processor(image, return_tensors="pt")
                out = blip_model.generate(**inputs)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)

            # 가장 높은 점수를 가진 이미지 선택
            selected_index = probs[0].index(max(probs[0]))
            selected_image = images[selected_index]

            # GPT-4를 사용하여 결과 설명 생성
            prompt_template = PromptTemplate(
                input_variables=["user_prompt", "image1_caption", "image2_caption", "selected_image", "reasoning"],
                template="""
                사용자 프롬프트: {user_prompt}

                이미지 1 캡션: {image1_caption}
                이미지 2 캡션: {image2_caption}

                선택된 이미지: {selected_image}

                선택 이유:
                {reasoning}
                """
            )

            # GPT-4로 설명 생성
            reasoning_input = {
                "user_prompt": user_prompt,
                "image1_caption": captions[0],
                "image2_caption": captions[1],
                "selected_image": f"이미지 {selected_index + 1}",
                "reasoning": (
                    f"CLIP 모델 분석 결과, 입력된 프롬프트 '{user_prompt}'와 가장 높은 유사도를 가진 이미지는 "
                    f"이미지 {selected_index + 1}입니다. 이 이미지는 프롬프트에 명시된 특징을 더 잘 반영합니다."
                ),
            }
            explanation = llm.predict(prompt_template.format(**reasoning_input))

        # 결과 출력
        st.success("분석 완료!")
        st.image(selected_image, caption=f"선택된 이미지 (이미지 {selected_index + 1})", use_column_width=True)
        st.write("## 분석 결과")
        st.write(f"**이미지 1 캡션:** {captions[0]}")
        st.write(f"**이미지 2 캡션:** {captions[1]}")
        st.write("## GPT-4 설명")
        st.write(explanation)
else:
    st.warning("두 개의 이미지를 업로드해주세요.")
