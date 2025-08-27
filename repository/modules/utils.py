import uuid
import json
import logging
import chromadb
import os
import re
from typing import List, Dict, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetroDataIndexer:
    """
    Data indexer for Riyadh Metro Assistant
    """
    
    def __init__(self, persist_path='../repository/chromadb'):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')
        
        # Optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
    
    def index_all_data(self):
        """Index all data files"""
        logger.info("Starting data indexing for Riyadh Metro Assistant")
        
        self._index_comprehensive_knowledge_base()
        self._index_amenities_data()
        
        logger.info("Completed indexing all data files")
    
    def _index_comprehensive_knowledge_base(self):
        """Index the comprehensive Riyadh Metro knowledge base"""
        logger.info("Indexing comprehensive Riyadh Metro knowledge base...")
        
        collection_name = "metro_rules"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/riyadh_knowledgebase.txt"
        if not os.path.exists(data_path):
            logger.warning(f"Knowledge base file not found: {data_path}")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = self._split_comprehensive_content(content)
        
        documents = []
        metadatas = []
        ids = []
        
        for section_title, section_content in sections:
            doc_content = f"Section: {section_title}\n\n{section_content}"
            
            documents.append(doc_content.strip())
            metadatas.append({
                "type": "metro_rules",
                "section": section_title,
                "category": "comprehensive_knowledge",
                "source": "riyadh_knowledgebase.txt"
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed knowledge base in {len(sections)} sections")
    
    def _split_comprehensive_content(self, content):
        """Split content into logical sections"""
        sections = []
        header_pattern = r'^##\s+(.+)$'
        lines = content.split('\n')
        
        current_section = ""
        current_title = "Overview"
        
        for line in lines:
            if re.match(header_pattern, line):
                if current_section.strip():
                    sections.append((current_title, current_section.strip()))
                current_title = re.match(header_pattern, line).group(1)
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section.strip():
            sections.append((current_title, current_section.strip()))
        
        return sections
    
    def _index_amenities_data(self):
        """Index amenities data"""
        logger.info("Indexing amenities data...")
        
        collection_name = "amenities"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/amenities.json"
        if not os.path.exists(data_path):
            logger.warning(f"Amenities file not found: {data_path}")
            return
        
        if os.path.getsize(data_path) == 0:
            logger.warning("Amenities file is empty")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            amenities_data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for amenity in amenities_data:
            title = amenity.get('title', 'Unknown')
            description = amenity.get('description', 'N/A')
            rating = amenity.get('rating', 'N/A')
            review = amenity.get('review', 'N/A')
            distance = amenity.get('Distance', 'N/A')
            walkable = amenity.get('Walkable', 'No')
            
            doc_content = f"""
            Location Name: {title}
            Description: {description}
            Rating: {rating}/5
            Customer Review: {review}
            Distance from KAFD: {distance}
            Walkable: {walkable}
            """
            
            category = self._categorize_amenity(title, description)
            
            documents.append(doc_content.strip())
            metadatas.append({
                "type": "amenities",
                "title": title,
                "rating": rating,
                "distance": distance,
                "walkable": walkable,
                "category": category,
                "search_terms": f"{title} {category} {description[:100]}"
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed {len(amenities_data)} amenities")
    
    def _categorize_amenity(self, title: str, description: str) -> str:
        """Categorize amenity based on title and description"""
        title_lower = title.lower()
        desc_lower = description.lower()
        
        if any(word in title_lower or word in desc_lower for word in ['restaurant', 'kitchen', 'bistro', 'grill']):
            return "Restaurants"
        elif any(word in title_lower or word in desc_lower for word in ['cafe', 'coffee', 'caffé', 'juice']):
            return "Cafes"
        elif any(word in title_lower or word in desc_lower for word in ['pharmacy', 'medical', 'health']):
            return "Healthcare"
        elif any(word in title_lower or word in desc_lower for word in ['hotel', 'residence', 'accommodation']):
            return "Accommodation"
        elif any(word in title_lower or word in desc_lower for word in ['park', 'garden', 'recreation']):
            return "Recreation"
        elif any(word in title_lower or word in desc_lower for word in ['mosque', 'prayer']):
            return "Religious"
        elif any(word in title_lower or word in desc_lower for word in ['store', 'shop', 'mall', 'retail']):
            return "Shopping"
        elif any(word in title_lower or word in desc_lower for word in ['bank', 'financial', 'office']):
            return "Business"
        elif any(word in title_lower or word in desc_lower for word in ['laundry', 'cleaning']):
            return "Services"
        elif any(word in title_lower or word in desc_lower for word in ['charging', 'ev', 'electric']):
            return "Transportation"
        else:
            return "Other"
    
    def _get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise
    
    def _add_to_collection(self, collection, ids: List[str], documents: List[str], metadatas: List[Dict]):
        """Add documents to collection"""
        try:
            # Clear existing data
            try:
                collection.delete(where={"type": {"$exists": True}})
            except Exception as e:
                logger.warning(f"Could not clear collection: {e}")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_model.embed_documents(documents)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings
                )
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            raise

def index_all_metro_data():
    """Main function to index all metro-related data"""
    try:
        indexer = MetroDataIndexer()
        indexer.index_all_data()
        logger.info("All metro data indexed successfully")
    except Exception as e:
        logger.error(f"Error during data indexing: {e}")
        raise

