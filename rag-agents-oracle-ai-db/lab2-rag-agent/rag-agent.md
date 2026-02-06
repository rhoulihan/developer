# Lab 2: Build a RAG Agent

## Introduction

In this lab, you'll build a complete Retrieval-Augmented Generation (RAG) agent using Oracle AI Database 26ai and OCI Generative AI. You'll learn how to process PDF documents, build retrieval functions, and create an interactive agent that can answer questions based on your knowledge base.

**Estimated Time:** 45 minutes

### Objectives

In this lab, you will:

- Load and process PDF documents from OCI Object Storage
- Chunk documents for effective retrieval
- Build a retrieval function that fetches relevant context
- Connect to OCI Generative AI chat models
- Implement a complete RAG pipeline
- Create an interactive agent with conversation history

### Prerequisites

This lab assumes you have:

- Completed Lab 1
- Familiarity with the concepts from Lab 1 (embeddings, vector search)

**Note:** The OCI Object Storage bucket with sample PDF is pre-configured in the sandbox.

## Task 1: Verify Object Storage Access

The sandbox environment has Object Storage credentials pre-configured. Let's verify access and check the available PDF document.

1. Start by verifying the Object Storage credential exists:

    ```python
    <copy>
    import oracledb
    import os

    # Re-establish connection (or continue from Lab 1)
    username = os.environ.get("DB_USER", "ADMIN")
    password = os.environ.get("DB_PASSWORD")
    dsn = os.environ.get("DB_DSN")

    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    # Verify Object Storage credential is available
    verify_cred_sql = """
    SELECT credential_name
    FROM user_credentials
    WHERE credential_name = 'OCI_OBJECT_CRED'
    """

    cursor.execute(verify_cred_sql)
    result = cursor.fetchone()

    if result:
        print(f"Object Storage credential '{result[0]}' is available")
    else:
        print("WARNING: OCI_OBJECT_CRED not found - contact lab administrator")

    # The PDF URI is available in the environment
    pdf_uri = os.environ.get('SAMPLE_PDF_URI')
    print(f"Sample PDF location: {pdf_uri}")
    </copy>
    ```

    > **Note:** The sandbox provides a pre-staged Oracle documentation PDF for this lab. The URI is available via the `SAMPLE_PDF_URI` environment variable.

    <!-- TODO: Add screenshot showing credentials verified -->
    ![Credentials Verified](../images/lab2/credentials-verified.png " ")

## Task 2: Load PDF Document

Now we'll load the Oracle documentation PDF from Object Storage into the database.

1. Create a table to store the PDF documents:

    ```python
    <copy>
    # Create table for PDF documents
    cursor.execute("""
        CREATE TABLE documents (
            id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            filename VARCHAR2(500),
            content BLOB,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.commit()
    print("Table 'documents' created successfully")
    </copy>
    ```

    If the table already exists, you can skip this step or handle the error:

    ```python
    <copy>
    try:
        cursor.execute("""
            CREATE TABLE documents (
                id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                filename VARCHAR2(500),
                content BLOB,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        print("Table 'documents' created successfully")
    except oracledb.DatabaseError as e:
        error, = e.args
        if error.code == 955:  # Table already exists
            print("Table 'documents' already exists")
        else:
            raise
    </copy>
    ```

2. Load the PDF from Object Storage:

    ```python
    <copy>
    # Load PDF from Object Storage (pre-configured in sandbox)
    load_pdf_sql = """
    DECLARE
        pdf_blob BLOB;
    BEGIN
        pdf_blob := DBMS_CLOUD.GET_OBJECT(
            credential_name => 'OCI_OBJECT_CRED',
            object_uri => :pdf_uri
        );

        INSERT INTO documents (filename, content)
        VALUES (:filename, pdf_blob);

        COMMIT;
    END;
    """

    # Get PDF URI from environment (pre-configured in sandbox)
    pdf_uri = os.environ.get('SAMPLE_PDF_URI')
    pdf_filename = pdf_uri.split('/')[-1] if pdf_uri else 'oracle-ai-vector-search-guide.pdf'

    try:
        cursor.execute(load_pdf_sql, {
            'pdf_uri': pdf_uri,
            'filename': pdf_filename
        })
        print(f"PDF document '{pdf_filename}' loaded successfully")
    except oracledb.DatabaseError as e:
        print(f"Note: {e}")
        print("Document may already be loaded")

    # Verify the document was loaded
    cursor.execute("SELECT id, filename, DBMS_LOB.GETLENGTH(content) as size_bytes FROM documents")
    print("\nLoaded Documents:")
    print("-" * 60)
    for row in cursor.fetchall():
        print(f"  ID: {row[0]}, File: {row[1]}, Size: {row[2]:,} bytes")
    </copy>
    ```

    <!-- TODO: Add screenshot of PDF loaded -->
    ![PDF Loaded](../images/lab2/pdf-loaded.png " ")

## Task 3: Process and Chunk Document

Now we'll extract text from the PDF, split it into chunks, and generate embeddings for each chunk.

1. Create a table to store the document chunks:

    ```python
    <copy>
    # Create table for document chunks with embeddings
    try:
        cursor.execute("""
            CREATE TABLE doc_chunks (
                id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                doc_id NUMBER,
                chunk_id NUMBER,
                chunk_text VARCHAR2(4000),
                chunk_vector VECTOR(1024, FLOAT32),
                CONSTRAINT fk_doc FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """)
        connection.commit()
        print("Table 'doc_chunks' created successfully")
    except oracledb.DatabaseError as e:
        error, = e.args
        if error.code == 955:
            print("Table 'doc_chunks' already exists")
        else:
            raise
    </copy>
    ```

2. Process the PDF - extract text, chunk, and generate embeddings:

    ```python
    <copy>
    # Process PDF: extract text, chunk, and generate embeddings
    # This is a powerful one-liner that does it all!

    print("Processing PDF document...")
    print("  - Extracting text with UTL_TO_TEXT")
    print("  - Chunking with UTL_TO_CHUNKS")
    print("  - Generating embeddings with UTL_TO_EMBEDDINGS")
    print("\nThis may take a few minutes...\n")

    process_document_sql = """
    INSERT INTO doc_chunks (doc_id, chunk_id, chunk_text, chunk_vector)
    SELECT
        d.id as doc_id,
        et.embed_id as chunk_id,
        et.embed_data as chunk_text,
        TO_VECTOR(et.embed_vector) as chunk_vector
    FROM documents d,
        TABLE(
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(
                    DBMS_VECTOR_CHAIN.UTL_TO_TEXT(d.content),
                    JSON('{"max_size": 1000, "overlap": 100, "normalize": "all"}')
                ),
                JSON('{
                    "provider": "ocigenai",
                    "credential_name": "OCI_GENAI_CRED",
                    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
                    "model": "cohere.embed-english-light-v3.0",
                    "input_type": "search_document"
                }')
            )
        ) et
    WHERE d.filename = :filename
    """

    cursor.execute(process_document_sql, {'filename': pdf_filename})
    connection.commit()

    # Verify chunks created
    cursor.execute("SELECT COUNT(*) FROM doc_chunks")
    chunk_count = cursor.fetchone()[0]
    print(f"Created {chunk_count} chunks from document")

    # Show sample chunks
    cursor.execute("""
        SELECT chunk_id, SUBSTR(chunk_text, 1, 100) as preview
        FROM doc_chunks
        WHERE ROWNUM <= 3
        ORDER BY chunk_id
    """)
    print("\nSample chunks:")
    print("-" * 60)
    for row in cursor.fetchall():
        print(f"Chunk {row[0]}: {row[1]}...")
    </copy>
    ```

    **Key points:**
    - `UTL_TO_TEXT()` extracts text from PDF (also works with Word docs, etc.)
    - `UTL_TO_CHUNKS()` splits text into ~1000 character chunks with 100 char overlap
    - `UTL_TO_EMBEDDINGS()` generates vectors for each chunk
    - Overlap ensures concepts aren't split across chunk boundaries

    <!-- TODO: Add screenshot of document processing -->
    ![Document Processed](../images/lab2/document-processed.png " ")

## Task 4: Create Vector Index for Chunks

Let's create a vector index on the document chunks for fast retrieval.

1. Run the following code to create the index:

    ```python
    <copy>
    # Create vector index on document chunks
    try:
        cursor.execute("""
            CREATE VECTOR INDEX doc_chunks_vec_idx ON doc_chunks(chunk_vector)
            ORGANIZATION INMEMORY NEIGHBOR GRAPH
            DISTANCE COSINE
            WITH TARGET ACCURACY 95
        """)
        print("Vector index 'doc_chunks_vec_idx' created successfully")
    except oracledb.DatabaseError as e:
        error, = e.args
        if "already exists" in str(error.message):
            print("Vector index already exists")
        else:
            raise

    # Verify index
    cursor.execute("""
        SELECT index_name, index_type
        FROM user_indexes
        WHERE table_name = 'DOC_CHUNKS' AND index_type LIKE '%VECTOR%'
    """)
    result = cursor.fetchone()
    if result:
        print(f"Index verified: {result[0]} ({result[1]})")
    </copy>
    ```

## Task 5: Build Retrieval Function

Now let's create a function that retrieves relevant context for a given query.

1. Create the retrieval function:

    ```python
    <copy>
    def retrieve_context(query: str, top_k: int = 5, max_tokens: int = 2000) -> str:
        """
        Retrieve relevant document chunks for a query.
        Returns concatenated context suitable for LLM input.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            max_tokens: Approximate token limit for context

        Returns:
            Concatenated text from most relevant chunks
        """

        sql = """
            SELECT chunk_text,
                   VECTOR_DISTANCE(chunk_vector,
                       (SELECT TO_VECTOR(et.embed_vector)
                        FROM TABLE(
                            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                                :query,
                                JSON('{
                                    "provider": "ocigenai",
                                    "credential_name": "OCI_GENAI_CRED",
                                    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
                                    "model": "cohere.embed-english-light-v3.0",
                                    "input_type": "search_query"
                                }')
                            )
                        ) et
                        WHERE ROWNUM = 1),
                       COSINE) as distance
            FROM doc_chunks
            ORDER BY distance
            FETCH APPROX FIRST :top_k ROWS ONLY WITH TARGET ACCURACY 90
        """

        cursor.execute(sql, {'query': query, 'top_k': top_k})
        results = cursor.fetchall()

        # Concatenate chunks, respecting token limit (approximate)
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars

        for chunk_text, distance in results:
            if total_chars + len(chunk_text) > char_limit:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    # Test retrieval
    test_query = "How do I create a vector index?"
    test_context = retrieve_context(test_query)
    print(f"Query: '{test_query}'")
    print(f"\nRetrieved context ({len(test_context)} chars):")
    print("-" * 60)
    print(test_context[:500] + "..." if len(test_context) > 500 else test_context)
    </copy>
    ```

    <!-- TODO: Add screenshot of retrieval test -->
    ![Retrieval Test](../images/lab2/retrieval-test.png " ")

## Task 6: Configure OCI GenAI Chat Model

Now let's set up the connection to OCI Generative AI for text generation.

1. Import the OCI SDK and create the client:

    ```python
    <copy>
    import oci
    from oci.generative_ai_inference import GenerativeAiInferenceClient
    from oci.generative_ai_inference.models import (
        ChatDetails,
        CohereChatRequest,
        OnDemandServingMode
    )

    # Initialize OCI client
    # In the sandbox, the config is pre-loaded
    config = oci.config.from_file()  # Uses ~/.oci/config or environment

    # TODO: Update endpoint if different region
    genai_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

    genai_client = GenerativeAiInferenceClient(
        config=config,
        service_endpoint=genai_endpoint
    )

    print("OCI Generative AI client initialized")
    print(f"Endpoint: {genai_endpoint}")
    </copy>
    ```

2. Create the response generation function:

    ```python
    <copy>
    def generate_response(prompt: str, model_id: str = "cohere.command-r-plus") -> str:
        """
        Generate a response using OCI Generative AI chat model.

        Args:
            prompt: The full prompt including context and question
            model_id: The model to use for generation

        Returns:
            Generated text response
        """

        chat_request = CohereChatRequest(
            message=prompt,
            max_tokens=1024,
            temperature=0.3,  # Lower = more focused/deterministic
            is_stream=False
        )

        chat_details = ChatDetails(
            compartment_id=os.environ.get('OCI_COMPARTMENT_OCID'),
            serving_mode=OnDemandServingMode(model_id=model_id),
            chat_request=chat_request
        )

        response = genai_client.chat(chat_details)
        return response.data.chat_response.text

    # Test LLM connection
    test_response = generate_response("Say 'Hello from OCI Generative AI!' in exactly those words.")
    print(f"LLM Test Response: {test_response}")
    </copy>
    ```

    <!-- TODO: Add screenshot of LLM test -->
    ![LLM Test](../images/lab2/llm-test.png " ")

## Task 7: Create RAG Pipeline

Now let's combine retrieval and generation into a complete RAG pipeline.

1. Create the RAG query function:

    ```python
    <copy>
    def rag_query(question: str, top_k: int = 5) -> str:
        """
        Complete RAG pipeline:
        1. Retrieve relevant context from the knowledge base
        2. Construct a prompt with the context
        3. Generate a response using the LLM

        Args:
            question: The user's question
            top_k: Number of context chunks to retrieve

        Returns:
            AI-generated answer grounded in the retrieved context
        """

        # Step 1: Retrieve context
        context = retrieve_context(question, top_k=top_k)

        # Step 2: Construct RAG prompt
        prompt = f"""You are a helpful assistant that answers questions about Oracle Database and AI Vector Search.

    Use the following context to answer the question. If the context doesn't contain relevant information, say so clearly. Do not make up information.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    Provide a clear, accurate answer based on the context above. If citing specific information, indicate where it comes from in the context."""

        # Step 3: Generate response
        response = generate_response(prompt)

        return response

    # Test RAG pipeline
    test_question = "What is the difference between HNSW and IVF vector indexes?"
    print(f"Question: {test_question}")
    print("\nGenerating RAG response...")
    print("-" * 60)
    answer = rag_query(test_question)
    print(f"\nAnswer:\n{answer}")
    </copy>
    ```

2. Try a few more questions:

    ```python
    <copy>
    # Test with different questions
    questions = [
        "How do I create a vector index in Oracle Database?",
        "What embedding models does Oracle support?",
        "How can I improve the accuracy of my similarity searches?"
    ]

    print("Testing RAG Pipeline with Multiple Questions")
    print("=" * 70)

    for q in questions:
        print(f"\nQ: {q}")
        print("-" * 70)
        answer = rag_query(q)
        # Truncate long answers for display
        if len(answer) > 300:
            print(f"A: {answer[:300]}...")
        else:
            print(f"A: {answer}")
    </copy>
    ```

    <!-- TODO: Add screenshot of RAG responses -->
    ![RAG Responses](../images/lab2/rag-responses.png " ")

## Task 8: Build Interactive Agent Loop

Let's create a more sophisticated agent with conversation history.

1. Create the RAGAgent class:

    ```python
    <copy>
    class RAGAgent:
        """
        RAG agent with conversation history support.

        The agent maintains context across multiple turns, allowing for
        follow-up questions and conversational interactions.
        """

        def __init__(self, history_turns: int = 3):
            """
            Initialize the RAG agent.

            Args:
                history_turns: Number of previous turns to include in context
            """
            self.history = []
            self.history_turns = history_turns

        def chat(self, user_input: str) -> str:
            """
            Process user input and generate a response.

            Args:
                user_input: The user's message or question

            Returns:
                AI-generated response
            """

            # Retrieve context based on current question
            context = retrieve_context(user_input, top_k=5)

            # Build conversation history string
            history_str = ""
            for h in self.history[-self.history_turns:]:
                history_str += f"User: {h['user']}\nAssistant: {h['assistant']}\n\n"

            # Construct prompt with history and context
            prompt = f"""You are a helpful assistant specializing in Oracle Database and AI Vector Search.

    CONVERSATION HISTORY:
    {history_str}

    RETRIEVED CONTEXT:
    {context}

    CURRENT QUESTION:
    {user_input}

    Provide a helpful, accurate response based on the context. If this is a follow-up question, consider the conversation history. If the context doesn't contain enough information to answer, say so clearly."""

            # Generate response
            response = generate_response(prompt)

            # Update history
            self.history.append({
                'user': user_input,
                'assistant': response
            })

            return response

        def clear_history(self):
            """Reset conversation history."""
            self.history = []
            print("Conversation history cleared.")

        def show_history(self):
            """Display conversation history."""
            if not self.history:
                print("No conversation history.")
                return

            print("Conversation History:")
            print("-" * 60)
            for i, h in enumerate(self.history, 1):
                print(f"\n[Turn {i}]")
                print(f"User: {h['user']}")
                print(f"Assistant: {h['assistant'][:200]}...")

    # Create agent instance
    agent = RAGAgent()
    print("RAG Agent initialized!")
    print("Commands: 'quit' to exit, 'clear' to reset history, 'history' to show history")
    </copy>
    ```

2. Test the agent with a conversation:

    ```python
    <copy>
    # Simulated conversation to demonstrate the agent
    conversation = [
        "What is Oracle AI Vector Search?",
        "How do I create an index for it?",
        "What's the difference between HNSW and IVF?"
    ]

    print("Demo Conversation with RAG Agent")
    print("=" * 60)

    for user_msg in conversation:
        print(f"\nYou: {user_msg}")
        response = agent.chat(user_msg)
        # Truncate for display
        display_response = response[:400] + "..." if len(response) > 400 else response
        print(f"\nAssistant: {display_response}")
        print("-" * 60)
    </copy>
    ```

3. (Optional) Run an interactive loop:

    ```python
    <copy>
    # Interactive loop - uncomment to run interactively
    # Note: This works best in a local Jupyter environment

    """
    print("\nStarting interactive mode...")
    print("Type your questions below. Commands: 'quit', 'clear', 'history'\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        elif user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear':
            agent.clear_history()
            continue
        elif user_input.lower() == 'history':
            agent.show_history()
            continue

        response = agent.chat(user_input)
        print(f"\nAssistant: {response}\n")
    """
    </copy>
    ```

    <!-- TODO: Add screenshot of agent conversation -->
    ![Agent Conversation](../images/lab2/agent-conversation.png " ")

## Task 9: Experiment with Different Queries

Test your RAG agent with various questions to see how it performs.

1. Run these test queries:

    ```python
    <copy>
    # Reset agent for fresh tests
    agent.clear_history()

    # Sample queries to test the RAG agent
    test_queries = [
        "How do I store vectors in Oracle Database?",
        "What is the VECTOR_DISTANCE function?",
        "Explain approximate nearest neighbor search.",
        "How can I combine keyword and vector search?",
        "What are the benefits of using Oracle for RAG applications?"
    ]

    print("Testing RAG Agent with Sample Queries")
    print("=" * 70)

    for query in test_queries:
        print(f"\nQ: {query}")
        response = rag_query(query)
        # Truncate for display
        print(f"A: {response[:300]}...")
        print("-" * 70)
    </copy>
    ```

2. Try edge cases:

    ```python
    <copy>
    # Test with a question the context might not cover
    edge_query = "What is the weather like today?"
    print(f"Edge Case Query: {edge_query}")
    print("-" * 60)
    response = rag_query(edge_query)
    print(f"Response: {response}")

    # The agent should indicate it doesn't have relevant information
    </copy>
    ```

    <!-- TODO: Add screenshot of query testing -->
    ![Query Testing](../images/lab2/query-testing.png " ")

## Task 10: Cleanup (Optional)

If you want to reset the lab, you can clean up the created objects:

```python
<copy>
# Optional: Clean up objects created in this lab
# Uncomment and run if you want to reset

# cursor.execute("DROP INDEX doc_chunks_vec_idx")
# cursor.execute("DROP TABLE doc_chunks PURGE")
# cursor.execute("DROP TABLE documents PURGE")
# connection.commit()
# print("Cleanup complete")
</copy>
```

## Summary

Congratulations! In this lab, you built a complete RAG agent! You learned how to:

- Load PDF documents from OCI Object Storage
- Process documents using `UTL_TO_TEXT`, `UTL_TO_CHUNKS`, and `UTL_TO_EMBEDDINGS`
- Build a retrieval function for fetching relevant context
- Connect to OCI Generative AI chat models
- Create a complete RAG pipeline that:
  - Retrieves relevant context based on the user's question
  - Constructs prompts that include the context
  - Generates accurate, grounded responses
- Build an interactive agent with conversation history

You now have the skills to build production RAG applications with Oracle AI Database 26ai!

## Quiz

```quiz score
Q: What function converts a PDF document to text in Oracle Database?
* DBMS_VECTOR_CHAIN.UTL_TO_TEXT
- DBMS_LOB.READ
- DBMS_CLOUD.GET_TEXT
- TO_CLOB
> UTL_TO_TEXT extracts text content from various document formats including PDF.

Q: What does RAG stand for?
* Retrieval-Augmented Generation
- Random Access Generation
- Rapid AI Generation
- Recursive Answer Generation
> RAG combines information retrieval with generative AI to produce more accurate, grounded responses.

Q: Why is document chunking important for RAG systems?
* Embedding models have token limits and smaller chunks improve retrieval precision
- It makes documents load faster
- It reduces storage costs
- It is required by Oracle Database
> Chunking enables embedding of large documents and improves retrieval by matching specific, relevant segments.

Q: What parameter controls the size of document chunks in UTL_TO_CHUNKS?
* max_size
- chunk_length
- token_limit
- segment_size
> The max_size parameter in the JSON configuration controls the maximum chunk size.

Q: Why is overlap between chunks useful?
* It prevents context from being split across chunk boundaries
- It reduces the total number of chunks
- It improves embedding model performance
- It is required for vector indexing
> Overlap ensures that concepts spanning chunk boundaries are captured in at least one complete chunk.

Q: What is the purpose of conversation history in a RAG agent?
* To maintain context across multiple turns of dialogue
- To store embeddings permanently
- To reduce API calls
- To cache document chunks
> Conversation history allows the agent to reference previous exchanges for coherent multi-turn conversations.

Q: Which OCI service provides the chat/LLM capabilities used in the RAG agent?
* OCI Generative AI
- OCI AI Services
- OCI Data Science
- OCI Functions
> OCI Generative AI provides both embedding models and chat/LLM models for RAG applications.

Q: What is "prompt engineering" in the context of RAG?
* Crafting effective prompts that include context and guide the LLM's response
- Writing code to call the API
- Optimizing database queries
- Training custom models
> Prompt engineering involves structuring the input to the LLM to produce the desired output format and quality.

Q: What happens if the retrieved context doesn't contain information relevant to the question?
* The LLM should acknowledge the limitation rather than hallucinate
- The system will crash
- The query is automatically retried
- The embedding model is retrained
> Good RAG implementations instruct the LLM to indicate when context is insufficient rather than making up information.

Q: What is the advantage of using Oracle Database for RAG compared to external vector databases?
* Data stays in one place with full SQL capabilities and transactional consistency
- External databases are always slower
- Oracle is the only database with vector support
- There is no advantage
> Oracle's converged database approach keeps vectors, documents, and relational data together with ACID guarantees.
```

## Learn More

- [Oracle AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/)
- [DBMS_VECTOR_CHAIN Package Reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_vector_chain1.html)
- [OCI Generative AI Documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- [Building RAG Applications with Oracle](https://www.oracle.com/artificial-intelligence/ai-vector-search/)

## Acknowledgements

- **Author** - Kirk Kirkconnell, Oracle
- **Contributors** - [Add contributors]
- **Last Updated By/Date** - [Your name], [Month Year]
