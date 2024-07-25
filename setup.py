from setuptools import setup, find_packages


setup(
    name="register-rag",
    packages=find_packages(),
    description="Quickly configure the RAG framework to meet your needs via JSON",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="JLULLM",
    author_email="none",
    url="https://github.com/Charon-ops/RegisterRAG",
    install_requires=[
        "chardet==5.2.0",
        "pypdf==4.2.0",
        "chromadb==0.5.3",
        "xinference-client==0.13.0",
        "transformers==4.42.3",
        "chardet==5.2.0",
        "sentence-transformers==3.0.1",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "torchvision==0.18.1",
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
