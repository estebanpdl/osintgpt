<div align="center">

# **OSINT GPT**

<br />

`osintgpt` is a Python package for leveraging OpenAI's GPT models to analyze text data and perform tasks such as calculating text embeddings, searching for similar documents, and more. It is designed for use in open-source intelligence (OSINT) applications and research.

<br />

<img src="https://raw.githubusercontent.com/estebanpdl/osintgpt/main/images/osintgpt.png" alt="osintgpt osint gpt" width="33%" height="33%" />

<br />
<br />

[![GitHub forks](https://img.shields.io/github/forks/estebanpdl/osintgpt.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/estebanpdl/osintgpt/network/)
[![GitHub stars](https://img.shields.io/github/stars/estebanpdl/osintgpt?style=social)](https://github.com/estebanpdl/osintgpt/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/estebanpdl/osintgpt/blob/main/LICENCE)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://twitter.com/estebanpdl)
[![Made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Twitter estebanpdl](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/estebanpdl)

</div>

<hr />
<br />

## **Installation**

You can install the `osint-gpt` package using pip:

```bash
pip install osintgpt
```

<hr />
<br />

## 🚀 **Features**

The `osintgpt` Python package is designed to streamline the process of analyzing text data by leveraging OpenAI's GPT models. Here are some of the key features:

- **Text Analysis**: Utilize OpenAI's GPT models to analyze text data, including calculating text embeddings and searching for similar documents.
- **Interactive Mode**: The package includes an interactive mode that allows users to communicate directly with the GPT model. The user can input a prompt and receive a response from the model, facilitating a more dynamic interaction.
- **Database Management**: The package integrates with SQLite database, enabling easy storage and retrieval of conversation data. The SQLDatabaseManager class creates tables, handles data insertion, and manages transactions.

Please note that the development of `osintgpt` is still in progress, and some features may still be refined or expanded.

<hr />
<br />


## 💾 **Vector store**

<h2>Qdrant</h2>

The `Qdrant` class is an interface to Qdrant, a high-performance vector similarity search engine. It provides a variety of methods for connecting and interacting with a Qdrant server, such as creating, updating, and deleting collections, and managing vector embeddings along with their associated payloads.

<h3>Main Features:</h3>

- **Connection Management**: The class allows you to establish and manage connections to a Qdrant server. The server can be accessed remotely or locally.
- **Collection Management**: You can create, update, and delete collections in Qdrant. Each collection can contain multiple vectors.
- **Vector and Payload Management**: The class provides methods to add, update, and search for vector embeddings in collections. Each vector can optionally have an associated payload. The payload represents data associated with the vector, such as metadata or additional features.
- **High Efficiency**: With the ability to efficiently store and search embeddings, Qdrant can support high-dimensional data and large-scale databases.

<h3>Setting Up Qdrant:</h3>

To use the Qdrant class, you will need access to a Qdrant server, either remotely or locally:

- **Remote Server**: Register for a remote server on [Qdrant Cloud](https://cloud.qdrant.io/).
- **Local Server**: Set up a local server following the instructions on the [Qdrant Quick Start guide](https://qdrant.tech/documentation/quick_start/).


<hr />
<br />


## **Disclaimer**

The `osintgpt` tool is provided for research purposes and intended to assist users in analyzing data from open-source intelligence (OSINT) tools more efficiently. It relies on third-party services, such as the OpenAI API, various database engines, and other resources that may have associated costs. By using this tool, you acknowledge that you are responsible for understanding and managing any costs related to these services. The creators and maintainers of `osintgpt` are not liable for any expenses incurred or any misuse of the tool. Please use this tool responsibly and in compliance with all applicable laws and regulations.

<hr />
<br />
