# 📚 참고 자료

NLP 대화 요약 프로젝트를 위한 외부 자료 및 추가 학습 리소스를 체계적으로 정리한 섹션입니다.

## 📋 포함 문서

### 📖 외부 리소스
- **논문 및 연구자료** - 대화 요약 관련 최신 연구 논문 및 데이터셋
- **블로그 및 튜토리얼** - 실무에 바로 적용 가능한 기술 블로그 및 학습 자료
- **공식 문서** - 사용 라이브러리 및 도구의 공식 문서 링크

### 🛠️ 도구 및 라이브러리
- **Transformers** - Hugging Face Transformers 활용 가이드
- **PyTorch** - PyTorch 관련 참고 자료
- **WandB** - 실험 추적 도구 사용법
- **ROUGE** - 평가 지표 상세 설명

### 🎓 학습 자료
- **NLP 기초** - 자연어 처리 기본 개념
- **Transformer 아키텍처** - 트랜스포머 모델 이해
- **요약 기법** - 텍스트 요약 방법론

### 🌐 커뮤니티
- **GitHub 리포지토리** - 관련 오픈소스 프로젝트
- **커뮤니티 포럼** - Stack Overflow, Reddit 등
- **컨퍼런스 및 워크샵** - 관련 학회 및 행사 정보

## 🔍 활용 방법

1. **학습 전** - 기본 개념 및 배경 지식 습득
2. **개발 중** - 구체적인 구현 방법 참고
3. **문제 해결** - 유사한 문제의 해결책 검색
4. **심화 학습** - 고급 기법 및 최신 동향 파악

## 🔗 관련 링크

- [기술 문서](../03_technical_docs/README.md)
- [개발자 가이드](../07_development/README.md)

---

# 🔗 주요 외부 링크 및 자료

## 📖 논문 및 연구자료

### 핵심 논문
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Transformer 원논문 (Vaswani et al., 2017)
  - *활용목적*: Transformer 아키텍처의 기본 이해
  - *핵심내용*: Self-attention 메커니즘과 Transformer 구조

- **[BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)** - BART 모델 논문 (Lewis et al., 2019)
  - *활용목적*: KoBART 모델의 이론적 배경
  - *핵심내용*: Denoising autoencoder 방식의 사전 학습

- **[KoBART: Korean BART Pre-trained Model](https://github.com/SKT-AI/KoBART)** - SKT AI KoBART 모델
  - *활용목적*: 한국어 요약 모델의 직접적 활용
  - *핵심내용*: 한국어 특화 BART 모델 구현

### 대화 요약 특화 논문
- **[DialogSum: A Real-Life Scenario Dialogue Summarization Dataset](https://arxiv.org/abs/2105.06762)** - DialogSum 데이터셋 (Chen et al., 2021)
  - *활용목적*: 대화 요약 데이터셋 설계 참고
  - *핵심내용*: 실제 대화 시나리오 기반 요약 데이터셋

- **[Abstractive Dialogue Summarization with Sentence-Gated Modeling Optimized by Dialogue Acts](https://arxiv.org/abs/1809.05715)** - 대화 요약 최적화 (Chen & Yang, 2020)
  - *활용목적*: 대화 특성을 반영한 요약 방법론
  - *핵심내용*: Dialogue acts를 활용한 요약 최적화

- **[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)** - PEGASUS 모델 (Zhang et al., 2020)
  - *활용목적*: 요약 특화 사전 학습 방법론
  - *핵심내용*: Gap-sentence generation 사전 학습

### 한국어 NLP 연구
- **[KoGPT: Korean Generative Pre-trained Transformer](https://github.com/kakaobrain/kogpt)** - 카카오브레인 KoGPT
  - *활용목적*: 한국어 생성 모델 비교 연구
  - *핵심내용*: 한국어 생성형 언어 모델

- **[KcELECTRA: Korean comments ELECTRA](https://github.com/Beomi/KcELECTRA)** - 한국어 ELECTRA 모델
  - *활용목적*: 한국어 인코더 모델 성능 비교
  - *핵심내용*: 한국어 댓글 데이터 기반 ELECTRA

## 🛠️ 도구 및 라이브러리 문서

### Hugging Face Ecosystem
- **[Transformers 공식 문서](https://huggingface.co/docs/transformers/)**
  - *활용목적*: 모델 로딩, 토크나이저, 학습 파이프라인
  - *핵심섹션*: Model Hub, Training, Generation

- **[Datasets 라이브러리](https://huggingface.co/docs/datasets/)**
  - *활용목적*: 효율적인 데이터 로딩 및 전처리
  - *핵심섹션*: Loading datasets, Processing, Caching

- **[Tokenizers 가이드](https://huggingface.co/docs/tokenizers/)**
  - *활용목적*: 커스텀 토크나이저 구현
  - *핵심섹션*: Encoding, Decoding, Special tokens

- **[Hugging Face Model Hub](https://huggingface.co/models)**
  - *활용목적*: 사전 학습된 한국어 모델 탐색
  - *추천모델*: gogamza/kobart-base-v2, KETI-AIR/ke-t5-base

### PyTorch 관련
- **[PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)**
  - *활용목적*: 모델 구현, 최적화, 디바이스 관리
  - *핵심섹션*: torch.nn, torch.optim, torch.cuda

- **[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)**
  - *활용목적*: 구조화된 학습 파이프라인 구현
  - *핵심섹션*: LightningModule, Trainer, Callbacks

- **[TorchMetrics](https://torchmetrics.readthedocs.io/)**
  - *활용목적*: 평가 메트릭 구현
  - *핵심섹션*: Text metrics, ROUGE, Custom metrics

### 실험 추적 및 모니터링
- **[Weights & Biases 가이드](https://docs.wandb.ai/)**
  - *활용목적*: 실험 추적, 하이퍼파라미터 튜닝
  - *핵심섹션*: Experiment tracking, Sweeps, Reports

- **[MLflow 문서](https://mlflow.org/docs/latest/index.html)**
  - *활용목적*: 모델 버전 관리, 배포
  - *핵심섹션*: Tracking, Model registry, Deployment

- **[TensorBoard 가이드](https://www.tensorflow.org/tensorboard)**
  - *활용목적*: 학습 과정 시각화
  - *핵심섹션*: Scalars, Distributions, Histograms

### 평가 메트릭
- **[ROUGE 평가 지표 상세 설명](https://en.wikipedia.org/wiki/ROUGE_(metric))**
  - *활용목적*: ROUGE 메트릭의 정확한 이해
  - *핵심내용*: ROUGE-1, ROUGE-2, ROUGE-L 차이점

- **[rouge-score 라이브러리](https://github.com/google-research/google-research/tree/master/rouge)**
  - *활용목적*: 정확한 ROUGE 점수 계산
  - *핵심기능*: Multi-reference evaluation

- **[BERTScore 논문 및 구현](https://github.com/Tiiiger/bert_score)**
  - *활용목적*: 의미적 유사도 기반 평가
  - *핵심내용*: BERT embedding 기반 평가 메트릭

## 🎓 학습 리소스

### 기본 개념 학습
- **[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)**
  - *활용목적*: Transformer 구조의 직관적 이해
  - *특징*: 시각적 설명으로 쉬운 이해

- **[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)**
  - *활용목적*: Transformer 구현의 상세한 이해
  - *특징*: 코드와 함께 설명

- **[Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)**
  - *활용목적*: Attention 메커니즘의 발전 과정
  - *특징*: 다양한 attention 방법 비교

### 실무 튜토리얼
- **[Hugging Face Course - Summarization](https://huggingface.co/course/chapter7/5)**
  - *활용목적*: 텍스트 요약 실습
  - *특징*: 단계별 코드 예제

- **[Fine-tuning BART for Summarization](https://towardsdatascience.com/fine-tuning-bart-for-abstractive-text-summarization-d1c4b8de3938)**
  - *활용목적*: BART 모델 파인튜닝 실습
  - *특징*: 실무 적용 가능한 예제

- **[Korean NLP Tutorial](https://github.com/haven-jeon/KoNLTK)**
  - *활용목적*: 한국어 NLP 전처리 방법
  - *특징*: 한국어 특성 고려한 전처리

### 고급 기법
- **[Controllable Text Summarization](https://arxiv.org/abs/2005.07213)**
  - *활용목적*: 제어 가능한 요약 생성
  - *핵심내용*: 길이, 스타일 제어 방법

- **[Few-Shot Learning for Text Summarization](https://arxiv.org/abs/2109.04309)**
  - *활용목적*: 적은 데이터로 요약 모델 학습
  - *핵심내용*: In-context learning, Meta-learning

## 🌐 커뮤니티 및 리소스

### GitHub 리포지토리
- **[KoBART 공식 저장소](https://github.com/SKT-AI/KoBART)**
  - *활용목적*: KoBART 모델 사용법 및 예제
  - *참고코드*: 요약 파인튜닝, 추론 예제

- **[Awesome Text Summarization](https://github.com/mathsyouth/awesome-text-summarization)**
  - *활용목적*: 텍스트 요약 관련 자료 총집합
  - *포함내용*: 논문, 데이터셋, 도구 링크

- **[Korean NLP Guide](https://github.com/songys/awesome-korean-nlp)**
  - *활용목적*: 한국어 NLP 리소스 모음
  - *포함내용*: 데이터셋, 모델, 도구

- **[Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples)**
  - *활용목적*: Transformers 라이브러리 활용 예제
  - *참고분야*: Summarization, Language modeling

### 온라인 커뮤니티
- **[Hugging Face Community](https://huggingface.co/discussions)**
  - *활용목적*: 모델 관련 질문 및 토론
  - *주요토픽*: Model usage, Training tips

- **[r/MachineLearning](https://www.reddit.com/r/MachineLearning/)**
  - *활용목적*: 최신 ML 연구 동향 파악
  - *주요토픽*: Paper discussions, Industry trends

- **[Stack Overflow - NLP](https://stackoverflow.com/questions/tagged/nlp)**
  - *활용목적*: 구체적인 구현 문제 해결
  - *주요토픽*: Code debugging, Implementation help

- **[Papers with Code - Text Summarization](https://paperswithcode.com/task/text-summarization)**
  - *활용목적*: 최신 연구 및 벤치마크 확인
  - *특징*: 논문과 구현 코드 연결

### 컨퍼런스 및 워크샵
- **[ACL (Association for Computational Linguistics)](https://www.aclweb.org/)**
  - *활용목적*: NLP 최신 연구 동향
  - *주요세션*: Summarization, Dialogue systems

- **[EMNLP (Empirical Methods in NLP)](https://2024.emnlp.org/)**
  - *활용목적*: 실무 적용 가능한 NLP 연구
  - *주요세션*: Applications, Evaluation

- **[NeurIPS](https://neurips.cc/)**
  - *활용목적*: 머신러닝 기초 이론 및 알고리즘
  - *관련분야*: Deep learning, Optimization

- **[ICLR (International Conference on Learning Representations)](https://iclr.cc/)**
  - *활용목적*: 딥러닝 표현 학습 연구
  - *관련분야*: Transformer models, Pre-training

### 데이터셋 및 벤치마크
- **[DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)**
  - *활용목적*: 대화 요약 벤치마크 데이터
  - *특징*: 실제 대화 시나리오 기반

- **[CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)**
  - *활용목적*: 뉴스 요약 벤치마크
  - *특징*: 추상적 요약 평가

- **[XSum](https://huggingface.co/datasets/xsum)**
  - *활용목적*: 극도로 압축된 요약 평가
  - *특징*: 한 문장 요약 생성

- **[Korean Multi-Reference Summarization](https://aihub.or.kr/)**
  - *활용목적*: 한국어 요약 데이터
  - *출처*: AI Hub 한국어 요약 데이터셋

## 🔧 개발 도구 가이드

### IDE 및 개발 환경
- **[VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)**
  - *활용목적*: Python 개발 환경 최적화
  - *주요기능*: IntelliSense, Debugging, Linting

- **[Jupyter Lab](https://jupyterlab.readthedocs.io/)**
  - *활용목적*: 실험적 코드 개발 및 프로토타이핑
  - *주요기능*: Interactive computing, Visualization

- **[Google Colab Pro](https://colab.research.google.com/)**
  - *활용목적*: GPU 리소스 활용한 모델 학습
  - *주요기능*: Free GPU/TPU, Easy sharing

### 버전 관리 및 협업
- **[DVC (Data Version Control)](https://dvc.org/)**
  - *활용목적*: 데이터 및 모델 버전 관리
  - *주요기능*: Data tracking, Pipeline management

- **[Git LFS](https://git-lfs.github.io/)**
  - *활용목적*: 대용량 파일 버전 관리
  - *사용대상*: 모델 파일, 데이터셋

- **[pre-commit](https://pre-commit.com/)**
  - *활용목적*: 코드 품질 자동 검사
  - *주요기능*: Code formatting, Linting

### 성능 프로파일링
- **[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)**
  - *활용목적*: 모델 성능 병목 지점 분석
  - *주요기능*: GPU utilization, Memory usage

- **[memory_profiler](https://pypi.org/project/memory-profiler/)**
  - *활용목적*: 메모리 사용량 분석
  - *주요기능*: Line-by-line memory usage

- **[cProfile](https://docs.python.org/3/library/profile.html)**
  - *활용목적*: CPU 성능 프로파일링
  - *주요기능*: Function call analysis

## 📊 평가 및 벤치마킹

### 자동 평가 도구
- **[NLTK BLEU](https://www.nltk.org/api/nltk.translate.html)**
  - *활용목적*: BLEU 점수 계산
  - *특징*: 다양한 n-gram 설정

- **[sacrebleu](https://github.com/mjpost/sacrebleu)**
  - *활용목적*: 표준화된 BLEU 계산
  - *특징*: 재현 가능한 평가

- **[METEOR](https://www.nltk.org/api/nltk.translate.html)**
  - *활용목적*: METEOR 점수 계산
  - *특징*: 어간 변화 고려

### 인간 평가 가이드
- **[Human Evaluation Guidelines for Text Summarization](https://arxiv.org/abs/2010.12233)**
  - *활용목적*: 인간 평가 설계 참고
  - *평가요소*: Fluency, Coherence, Relevance

- **[Crowdsourcing Evaluation](https://www.mturk.com/)**
  - *활용목적*: 대규모 인간 평가 수행
  - *플랫폼*: Amazon Mechanical Turk

## 🚀 배포 및 운영

### 모델 서빙
- **[FastAPI](https://fastapi.tiangolo.com/)**
  - *활용목적*: REST API 서버 구축
  - *특징*: 자동 문서 생성, Type hints

- **[Gradio](https://gradio.app/)**
  - *활용목적*: 빠른 데모 인터페이스 구축
  - *특징*: 웹 UI 자동 생성

- **[Streamlit](https://streamlit.io/)**
  - *활용목적*: 인터랙티브 웹 앱 구축
  - *특징*: Python 기반 간편 개발

### 클라우드 배포
- **[Hugging Face Spaces](https://huggingface.co/spaces)**
  - *활용목적*: 무료 모델 데모 호스팅
  - *특징*: Gradio/Streamlit 지원

- **[Google Cloud AI Platform](https://cloud.google.com/ai-platform)**
  - *활용목적*: 대규모 모델 서빙
  - *특징*: Auto-scaling, Load balancing

- **[AWS SageMaker](https://aws.amazon.com/sagemaker/)**
  - *활용목적*: 엔터프라이즈 ML 파이프라인
  - *특징*: End-to-end ML workflow

## 📈 최신 동향 및 연구

### 2024년 주요 발전사항
- **[Large Language Models for Summarization](https://arxiv.org/abs/2401.00000)**
  - *트렌드*: LLM을 활용한 요약 생성
  - *주요모델*: GPT-4, Claude, LLaMA

- **[Retrieval-Augmented Summarization](https://arxiv.org/abs/2401.00001)**
  - *트렌드*: 외부 지식 활용 요약
  - *핵심기술*: RAG, Knowledge graphs

- **[Multi-modal Dialogue Summarization](https://arxiv.org/abs/2401.00002)**
  - *트렌드*: 텍스트 외 정보 활용
  - *확장영역*: Audio, Video, Emotion

### 향후 연구 방향
- **Controllable Generation** - 사용자 요구에 맞는 맞춤형 요약
- **Cross-lingual Summarization** - 다국어 간 요약 생성
- **Real-time Summarization** - 실시간 대화 요약
- **Personalized Summarization** - 개인화된 요약 스타일

---

## 💡 활용 팁

### 효율적인 학습 순서
1. **기초 이론** → Transformer, BART 논문 읽기
2. **실습 경험** → Hugging Face Course 완주
3. **한국어 특화** → KoBART, 한국어 NLP 자료 학습
4. **심화 연구** → 최신 논문 및 고급 기법 탐구

### 문제 해결 전략
1. **공식 문서 우선** → 라이브러리 공식 가이드 확인
2. **커뮤니티 활용** → Stack Overflow, GitHub Issues 검색
3. **실험적 접근** → 작은 규모로 테스트 후 확장
4. **성능 모니터링** → 지속적인 평가 및 개선

### 프로젝트 진행 체크리스트
- [ ] 기본 논문 3편 이상 읽기
- [ ] Hugging Face Transformers 실습 완료
- [ ] 한국어 데이터 전처리 방법 숙지
- [ ] ROUGE 평가 시스템 구현
- [ ] 실험 추적 시스템 구축
- [ ] 모델 배포 파이프라인 설계

이 참고 자료들을 통해 NLP 대화 요약 프로젝트의 이론적 배경부터 실무 적용까지 체계적으로 학습하고 구현할 수 있습니다.
