from setuptools import setup, find_packages


setup(
    name="register-rag",
    packages=find_packages(),
    description="Quickly configure the RAG framework to meet your needs via JSON",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="123",
    author_email="none",
    url="https://github.com/Charon-ops/RegisterRAG",
    install_requires=[
        "langchain==0.2.1",
        "fastapi==0.111.0",
        "uvicorn==0.29.0",
        "arxiv==2.1.0",
        "beautifulsoup4==4.12.3",
        "langchain-community==0.2.1",
        "chardet==5.2.0",
        "brotli==1.1.0",
        "scipy==1.13.1",
        "spaCy==3.7.5",
        "datasketch==1.6.5",
        "pypdf==4.2.0",
        "dashscope==1.19.2",
        "chromadb==0.5.3",
        "xinference-client==0.13.0",
        "gradio==4.37.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Operating System :: OS Independent",
    ],
)
