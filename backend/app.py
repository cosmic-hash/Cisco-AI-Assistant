from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import uuid
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np
from typing import List, Dict
import re
import io
import pdfplumber
import pandas as pd
from docx import Document

load_dotenv()

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

# Environment variables
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB', 'ragdb')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'chunks')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
GEN_TEMPERATURE = float(os.getenv('GEN_TEMPERATURE', '0.2'))
TOP_K = int(os.getenv('TOP_K', '5'))

# Initialize models
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Embedding model loaded!")

# Initialize Gemini client
print("🔄 Initializing Gemini API client...")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
    print("   Set it in your .env file or as an environment variable")
    gemini_client = None
else:
    try:
        # Set the API key as environment variable for the client
        os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✅ Gemini API client initialized with model: {GEMINI_MODEL}")
    except Exception as e:
        print(f"❌ Error initializing Gemini client: {e}")
        gemini_client = None

# MongoDB connection
client = None
db = None
collection = None

if MONGODB_URI:
    try:
        print(f"🔄 Connecting to MongoDB Atlas...")
        client = MongoClient(
            MONGODB_URI, 
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=10000  # 10 second timeout
        )
        # Test connection
        client.admin.command('ping')
        db = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        print("✅ MongoDB connection established!")
        print(f"   Database: {MONGODB_DB}")
        print(f"   Collection: {MONGODB_COLLECTION}")
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check if 'cluster0.mongodb.net' is your actual cluster hostname")
        print("   2. Verify your IP is whitelisted in MongoDB Atlas")
        print("   3. Check your username and password are correct")
        print("   4. Ensure your cluster is running")
        client = None
        db = None
        collection = None
else:
    print("⚠️ MONGODB_URI not set in environment variables")


def extract_text_from_file(file, filename: str) -> str:
    """Extract text content from various file types."""
    file_extension = filename.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        # Extract text from PDF
        text_content = []
        try:
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            return '\n\n'.join(text_content)
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    elif file_extension in ['xlsx', 'xls']:
        # Extract text from Excel
        try:
            file.seek(0)  # Reset file pointer
            df = pd.read_excel(io.BytesIO(file.read()), sheet_name=None)
            text_content = []
            for sheet_name, sheet_df in df.items():
                text_content.append(f"Sheet: {sheet_name}\n")
                # Convert DataFrame to string representation
                text_content.append(sheet_df.to_string(index=False))
                text_content.append("\n")
            return '\n'.join(text_content)
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")
    
    elif file_extension == 'docx':
        # Extract text from Word document
        try:
            file.seek(0)
            doc = Document(io.BytesIO(file.read()))
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            return '\n'.join(text_content)
        except Exception as e:
            raise Exception(f"Error reading Word document: {str(e)}")
    
    elif file_extension in ['txt', 'sql', 'py', 'js', 'html', 'css', 'json', 'yaml', 'yml', 'md']:
        # Plain text files
        file.seek(0)
        return file.read().decode('utf-8')
    
    else:
        raise Exception(f"Unsupported file type: .{file_extension}")


def chunk_text(text: str, source_id: str, file_type: str = 'text') -> List[Dict]:
    """Chunk text by SQL statements (;) or ~1000 characters."""
    chunks = []
    
    # Try to split by SQL statements first (for SQL files)
    if file_type == 'sql':
        sql_statements = re.split(r';\s*\n', text)
        
        if len(sql_statements) > 1:
            # SQL file - chunk by statements
            for i, statement in enumerate(sql_statements):
                statement = statement.strip()
                if statement:
                    # If statement is too long, split further
                    if len(statement) > 1000:
                        for j in range(0, len(statement), 1000):
                            chunk_text = statement[j:j+1000]
                            chunks.append({
                                'text': chunk_text,
                                'source_id': source_id,
                                'chunk_id': str(uuid.uuid4())
                            })
                    else:
                        chunks.append({
                            'text': statement,
                            'source_id': source_id,
                            'chunk_id': str(uuid.uuid4())
                        })
            return chunks
    
    # Regular text chunking - chunk by ~1000 characters with overlap
    chunk_size = 1000
    overlap = 200  # Overlap to maintain context
    
    i = 0
    while i < len(text):
        chunk_text = text[i:i+chunk_size]
        if chunk_text.strip():  # Only add non-empty chunks
            chunks.append({
                'text': chunk_text.strip(),
                'source_id': source_id,
                'chunk_id': str(uuid.uuid4())
            })
        i += chunk_size - overlap  # Move forward with overlap
    
    return chunks


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify MongoDB connection."""
    try:
        if client is None or collection is None:
            return jsonify({
                'status': 'error', 
                'message': 'MongoDB not connected',
                'mongodb_uri_set': bool(MONGODB_URI),
                'troubleshooting': {
                    'check_cluster_hostname': 'Verify cluster0.mongodb.net is your actual cluster name',
                    'check_ip_whitelist': 'Ensure your IP is whitelisted in MongoDB Atlas',
                    'check_credentials': 'Verify username and password are correct'
                }
            }), 500
        
        # Test connection
        client.admin.command('ping')
        
        # Get collection stats
        stats = db.command('collStats', MONGODB_COLLECTION)
        doc_count = collection.count_documents({})
        
        return jsonify({
            'status': 'healthy',
            'mongodb': 'connected',
            'database': MONGODB_DB,
            'collection': MONGODB_COLLECTION,
            'document_count': doc_count,
            'collection_size': stats.get('size', 0)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/api/files', methods=['GET'])
def get_uploaded_files():
    """Get list of all uploaded files from MongoDB."""
    try:
        if collection is None:
            return jsonify({'error': 'MongoDB not connected'}), 500
        
        # Aggregate to get unique source files with metadata
        pipeline = [
            {
                '$group': {
                    '_id': '$source_id',
                    'chunks_count': {'$sum': 1},
                    'file_type': {'$first': '$metadata.file_extension'},
                    'type': {'$first': '$metadata.type'},
                    'last_updated': {'$max': '$_id'}  # Use ObjectId timestamp as proxy
                }
            },
            {
                '$project': {
                    'source_id': '$_id',
                    'chunks_count': 1,
                    'file_type': 1,
                    'type': 1,
                    '_id': 0
                }
            },
            {
                '$sort': {'source_id': 1}
            }
        ]
        
        files = list(collection.aggregate(pipeline))
        
        return jsonify({
            'status': 'success',
            'files': files,
            'total_files': len(files)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ingest', methods=['POST'])
def ingest():
    """Ingest various file types (PDF, Excel, Word, text, SQL, etc.), chunk them, embed, and store in MongoDB."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = file.filename
    file_extension = filename.lower().split('.')[-1]
    
    # Supported file types
    supported_extensions = ['pdf', 'xlsx', 'xls', 'docx', 'txt', 'sql', 'py', 'js', 'html', 'css', 'json', 'yaml', 'yml', 'md']
    
    if file_extension not in supported_extensions:
        return jsonify({
            'error': f'Unsupported file type: .{file_extension}',
            'supported_types': supported_extensions
        }), 400
    
    try:
        # Extract text content from file
        content = extract_text_from_file(file, filename)
        
        if not content or not content.strip():
            return jsonify({'error': 'No text content found in file'}), 400
        
        source_id = filename
        file_type = 'sql' if file_extension == 'sql' else 'text'
        
        # Chunk the content
        chunks = chunk_text(content, source_id, file_type)
        
        if not chunks:
            return jsonify({'error': 'No chunks created from file content'}), 400
        
        # Embed and store chunks
        stored_count = 0
        for chunk in chunks:
            # Generate embedding
            embedding = embedding_model.encode(chunk['text']).tolist()
            
            # Determine metadata type
            type_mapping = {
                'pdf': 'pdf_spec',
                'xlsx': 'excel_spec',
                'xls': 'excel_spec',
                'docx': 'docx_spec',
                'sql': 'sql_spec',
                'txt': 'text_spec',
                'py': 'code_spec',
                'js': 'code_spec',
                'html': 'code_spec',
                'css': 'code_spec',
                'json': 'code_spec',
                'yaml': 'config_spec',
                'yml': 'config_spec',
                'md': 'markdown_spec'
            }
            
            # Prepare document
            document = {
                'source_id': chunk['source_id'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'metadata': {
                    'type': type_mapping.get(file_extension, 'text_spec'),
                    'file_extension': file_extension,
                    'purpose': 'config_generation'
                },
                'embedding': embedding
            }
            
            # Store in MongoDB
            collection.insert_one(document)
            stored_count += 1
        
        return jsonify({
            'status': 'success',
            'message': f'Ingested {stored_count} chunks from {filename}',
            'chunks': stored_count,
            'source_id': source_id,
            'file_type': file_extension
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_context():
    """Search for relevant context chunks in the database based on query text."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    top_k = int(data.get('top_k', TOP_K))
    
    try:
        print(f"\n{'='*60}", flush=True)
        print(f"🔍 SEARCH API - RETRIEVAL ONLY", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Query: {query}", flush=True)
        print(f"Top K: {top_k}", flush=True)
        
        # Generate embedding for the query
        print(f"🔄 Generating embedding for query...", flush=True)
        query_embedding = embedding_model.encode(query).tolist()
        print(f"✅ Embedding generated (dimension: {len(query_embedding)})", flush=True)
        sys.stdout.flush()
        
        # Vector search in MongoDB
        pipeline_search = [
            {
                '$vectorSearch': {
                    'index': 'idx_embeddings',
                    'path': 'embedding',
                    'queryVector': query_embedding,
                    'numCandidates': top_k * 10,
                    'limit': top_k
                }
            },
            {
                '$project': {
                    'text': 1,
                    'source_id': 1,
                    'chunk_id': 1,
                    'metadata': 1,
                    'score': {'$meta': 'vectorSearchScore'}
                }
            }
        ]
        
        # Try vector search, fallback to manual cosine similarity if index doesn't exist
        try:
            results = list(collection.aggregate(pipeline_search))
            print(f"✅ Vector search successful, found {len(results)} results", flush=True)
            sys.stdout.flush()
        except Exception as e:
            # Fallback: calculate cosine similarity manually
            print(f"⚠️ Vector search index not found, using manual cosine similarity", flush=True)
            print(f"Error: {e}", flush=True)
            
            all_docs = list(collection.find({}, {'embedding': 1, 'text': 1, 'source_id': 1, 'chunk_id': 1, 'metadata': 1}))
            print(f"📄 Total documents retrieved from DB: {len(all_docs)}", flush=True)
            
            if len(all_docs) == 0:
                print("❌ No documents found in collection!", flush=True)
                return jsonify({
                    'error': 'No documents found in database',
                    'suggestion': 'Upload and ingest files first'
                }), 404
            
            # Calculate cosine similarities
            query_vec = np.array(query_embedding)
            similarities = []
            docs_with_embeddings = 0
            
            for doc in all_docs:
                if 'embedding' in doc and doc['embedding']:
                    docs_with_embeddings += 1
                    doc_vec = np.array(doc['embedding'])
                    cosine_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                    similarities.append((cosine_sim, doc))
            
            print(f"📈 Documents with embeddings: {docs_with_embeddings}/{len(all_docs)}", flush=True)
            
            # Sort by similarity and take top K
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_sims = [f'{s[0]:.4f}' for s in similarities[:5]]
            print(f"🎯 Top similarities: {top_sims}", flush=True)
            
            results = [{'text': doc['text'], 'source_id': doc['source_id'], 
                       'chunk_id': doc['chunk_id'], 'metadata': doc.get('metadata', {}),
                       'score': float(sim)} for sim, doc in similarities[:top_k]]
            
            print(f"✅ Manual cosine similarity found {len(results)} results", flush=True)
            sys.stdout.flush()
        
        # Format results for response
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'rank': i + 1,
                'chunk_id': result.get('chunk_id', ''),
                'source_id': result.get('source_id', 'Unknown'),
                'text': result['text'],
                'text_preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                'score': float(result.get('score', 0)),
                'metadata': result.get('metadata', {})
            })
            
            # Print to console
            print(f"\n--- Result {i+1} ---", flush=True)
            print(f"Source: {result.get('source_id', 'Unknown')}", flush=True)
            print(f"Score: {result.get('score', 0):.4f}", flush=True)
            print(f"Text: {result['text'][:300]}...", flush=True)
        
        print(f"\n{'='*60}", flush=True)
        print(f"✅ Returning {len(formatted_results)} results", flush=True)
        print(f"{'='*60}\n", flush=True)
        sys.stdout.flush()
        
        return jsonify({
            'status': 'success',
            'query': query,
            'total_results': len(formatted_results),
            'results': formatted_results
        })
    
    except Exception as e:
        print(f"❌ Error in search: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-config', methods=['POST'])
def generate_config():
    """Generate configuration based on user instruction using RAG."""
    data = request.get_json()
    if not data or 'instruction' not in data:
        return jsonify({'error': 'No instruction provided'}), 400
    
    instruction = data['instruction']
    
    try:
        # Embed the instruction
        instruction_embedding = embedding_model.encode(instruction).tolist()
        
        # Vector search in MongoDB
        # Note: MongoDB Atlas vector search requires aggregation pipeline
        # For local testing, we'll use cosine similarity calculation
        pipeline_search = [
            {
                '$vectorSearch': {
                    'index': 'idx_embeddings',
                    'path': 'embedding',
                    'queryVector': instruction_embedding,
                    'numCandidates': TOP_K * 10,
                    'limit': TOP_K
                }
            },
            {
                '$project': {
                    'text': 1,
                    'source_id': 1,
                    'chunk_id': 1,
                    'metadata': 1,
                    'score': {'$meta': 'vectorSearchScore'}
                }
            }
        ]
        
        # Try vector search, fallback to manual cosine similarity if index doesn't exist
        print(f"\n{'='*60}", flush=True)
        print(f"🔍 SEARCHING FOR RELEVANT CONTEXT", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Instruction: {instruction}", flush=True)
        print(f"Top K: {TOP_K}", flush=True)
        doc_count = collection.count_documents({})
        print(f"Total documents in collection: {doc_count}", flush=True)
        sys.stdout.flush()
        
        try:
            results = list(collection.aggregate(pipeline_search))
            print(f"✅ Vector search successful, found {len(results)} results", flush=True)
            sys.stdout.flush()
        except Exception as e:
            # Fallback: calculate cosine similarity manually
            print(f"⚠️ Vector search index not found, using manual cosine similarity", flush=True)
            print(f"Error: {e}", flush=True)
            
            all_docs = list(collection.find({}, {'embedding': 1, 'text': 1, 'source_id': 1, 'chunk_id': 1, 'metadata': 1}))
            print(f"📄 Total documents retrieved from DB: {len(all_docs)}", flush=True)
            
            if len(all_docs) == 0:
                print("❌ No documents found in collection!", flush=True)
            else:
                print(f"📊 Sample document keys: {list(all_docs[0].keys()) if all_docs else 'N/A'}", flush=True)
            
            # Calculate cosine similarities
            instruction_vec = np.array(instruction_embedding)
            similarities = []
            docs_with_embeddings = 0
            
            for doc in all_docs:
                if 'embedding' in doc and doc['embedding']:
                    docs_with_embeddings += 1
                    doc_vec = np.array(doc['embedding'])
                    cosine_sim = np.dot(instruction_vec, doc_vec) / (np.linalg.norm(instruction_vec) * np.linalg.norm(doc_vec))
                    similarities.append((cosine_sim, doc))
            
            print(f"📈 Documents with embeddings: {docs_with_embeddings}/{len(all_docs)}", flush=True)
            
            # Sort by similarity and take top K
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_sims = [f'{s[0]:.4f}' for s in similarities[:5]]
            print(f"🎯 Top similarities: {top_sims}", flush=True)
            
            results = [{'text': doc['text'], 'source_id': doc['source_id'], 
                       'chunk_id': doc['chunk_id'], 'metadata': doc.get('metadata', {}),
                       'score': sim} for sim, doc in similarities[:TOP_K]]
            
            print(f"✅ Manual cosine similarity found {len(results)} results", flush=True)
            sys.stdout.flush()
        
        # Check if we have any results
        print(f"\n{'='*60}", flush=True)
        print(f"📋 RETRIEVAL RESULTS", flush=True)
        print(f"{'='*60}", flush=True)
        num_results = len(results) if results else 0
        print(f"Number of results: {num_results}", flush=True)
        sys.stdout.flush()
        
        if not results or len(results) == 0:
            print("❌ No results found!", flush=True)
            sys.stdout.flush()
            return jsonify({
                'error': 'No relevant context found in database. Please upload and ingest files first.',
                'suggestion': 'Upload some files using the /api/ingest endpoint before generating configurations.'
            }), 400
        
        # Print retrieved chunks to console
        print(f"\n{'='*60}", flush=True)
        print(f"📚 RETRIEVED CHUNKS FROM DATABASE ({len(results)} chunks)", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Build prompt with context
        context_chunks = []
        context_chunks_detailed = []  # For UI display
        
        for i, result in enumerate(results):
            chunk_text = f"<<<CHUNK {i+1}>>>\n{result['text']}"
            context_chunks.append(chunk_text)
            
            # Print chunk details to console
            print(f"\n--- Chunk {i+1} ---", flush=True)
            print(f"Source: {result.get('source_id', 'Unknown')}", flush=True)
            score_str = f"{result.get('score', 0):.4f}" if result.get('score') else "N/A"
            print(f"Score: {score_str}", flush=True)
            text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            print(f"Text preview: {text_preview}", flush=True)
            print(f"Full text length: {len(result['text'])} characters", flush=True)
            print(f"\n--- Full Chunk {i+1} Text ---", flush=True)
            print(result['text'], flush=True)
            print(f"--- End Chunk {i+1} ---\n", flush=True)
            
            # Store detailed info for UI
            context_chunks_detailed.append({
                'chunk_number': i + 1,
                'text': result['text'],
                'source_id': result.get('source_id', 'Unknown'),
                'score': result.get('score', 0),
                'metadata': result.get('metadata', {})
            })
        
        context_text = "\n\n".join(context_chunks)
        print(f"\n{'='*60}", flush=True)
        print(f"📝 CONTEXT TEXT LENGTH: {len(context_text)} characters", flush=True)
        print(f"{'='*60}", flush=True)
        sys.stdout.flush()

        current_topology  = """{
            "topology": {
                "description": "Two Cisco NCS 5000 routers with interconnection",
                "routers": [
                {
                    "id": "R1",
                    "hostname": "NCS5000-R1",
                    "model": "NCS-5500",
                    "management_ip": "10.0.0.1",
                    "loopback0": "192.168.1.1/32",
                    "interfaces": [
                    {
                        "name": "HundredGigE0/0/0/0",
                        "description": "Link to R2",
                        "ip_address": "10.1.1.1/30",
                        "status": "up",
                        "connected_to": {
                        "router": "R2",
                        "interface": "HundredGigE0/0/0/0"
                        }
                    },
                    {
                        "name": "MgmtEth0/RP0/CPU0/0",
                        "description": "Management Interface",
                        "ip_address": "10.0.0.1/24",
                        "status": "up"
                    }
                    ]
                },
                {
                    "id": "R2",
                    "hostname": "NCS5000-R2",
                    "model": "NCS-5500",
                    "management_ip": "10.0.0.2",
                    "loopback0": "192.168.1.2/32",
                    "interfaces": [
                    {
                        "name": "HundredGigE0/0/0/0",
                        "description": "Link to R1",
                        "ip_address": "10.1.1.2/30",
                        "status": "up",
                        "connected_to": {
                        "router": "R1",
                        "interface": "HundredGigE0/0/0/0"
                        }
                    },
                    {
                        "name": "MgmtEth0/RP0/CPU0/0",
                        "description": "Management Interface",
                        "ip_address": "10.0.0.2/24",
                        "status": "up"
                    }
                    ]
                }
                ],
                "links": [
                {
                    "id": "link-1",
                    "type": "point-to-point",
                    "bandwidth": "100G",
                    "endpoints": [
                    {
                        "router": "R1",
                        "interface": "HundredGigE0/0/0/0",
                        "ip": "10.1.1.1/30"
                    },
                    {
                        "router": "R2",
                        "interface": "HundredGigE0/0/0/0",
                        "ip": "10.1.1.2/30"
                    }
                    ],
                    "protocols": ["ISIS", "MPLS", "BGP"]
                }
                ],
                "routing": {
                "igp": "ISIS",
                "bgp": {
                    "as_number": 65000,
                    "neighbors": [
                    {
                        "router": "R1",
                        "peer": "192.168.1.2",
                        "remote_as": 65000
                    },
                    {
                        "router": "R2",
                        "peer": "192.168.1.1",
                        "remote_as": 65000
                    }
                    ]
                }
                }
            }
            }"""
        
        # ====================================================================
        # PROMPT CONSTRUCTION FOR LLM (Gemini API)
        # This is the exact prompt that gets sent to the LLM for generation
        # ====================================================================

        # Current topology:
# {current_topology}


        prompt = f"""You are a backend configuration-generation assistant.
You write valid, ready-to-use configuration files or code blocks  specifically for IOS-XR devices for NCS series
strictly following provided context. Do not invent unsupported fields.

CONTEXT:
{context_text}

USER INSTRUCTION:
{instruction}

Current topology:
{current_topology}


OUTPUT FORMAT- GENERATE CONFIGURATION:
- Return only the generated configuration/code inside proper fenced code blocks.
- Mention which context chunks you referenced.

If its a general explanation question, return the explanation based on the context added. Never use your own knowledge or experience to answer the question.
OUTPUT FORMAT- GENERATE EXPLANATION:
 - Return only the explanation.
"""
        # ====================================================================
        
        print(f"\n{'='*60}", flush=True)
        print(f"💬 PROMPT TO BE SENT TO LLM", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Prompt length: {len(prompt)} characters", flush=True)
        print(f"Context chunks in prompt: {len(context_chunks)}", flush=True)
        print(f"\n--- Full Prompt ---", flush=True)
        print(prompt, flush=True)
        print(f"\n--- End Prompt ---", flush=True)
        print(f"{'='*60}\n", flush=True)
        sys.stdout.flush()
        
        # Generate using Gemini API
        if not gemini_client:
            raise Exception("Gemini API client not initialized. Check your GEMINI_API_KEY.")
        
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": GEN_TEMPERATURE,
                }
            )
            generated_text = response.text
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")
        
        # Extract code blocks if present
        code_block_match = re.search(r'```[\w]*\n(.*?)```', generated_text, re.DOTALL)
        if code_block_match:
            config_output = code_block_match.group(1).strip()
        else:
            config_output = generated_text.strip()
        
        # Prepare references with full text
        references = [
            {
                'chunk_id': r.get('chunk_id', ''),
                'source_id': r.get('source_id', ''),
                'text_preview': r['text'][:200] + '...' if len(r['text']) > 200 else r['text'],
                'text_full': r['text'],  # Full text for display
                'score': r.get('score', 0),
                'metadata': r.get('metadata', {})
            }
            for r in results
        ]
        
        return jsonify({
            'config_output': config_output,
            'references': references,
            'full_response': generated_text,
            'prompt_used': prompt,  # Return the full prompt
            'context_chunks': context_chunks,  # Return all context chunks (formatted)
            'context_chunks_detailed': context_chunks_detailed,  # Return detailed chunks for UI
            'embedding_model': EMBEDDING_MODEL  # Return embedding model name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend."""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

