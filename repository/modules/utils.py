import uuid
import json
import logging
import chromadb
import os
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetroDataIndexer:
    """
    Comprehensive data indexer for Riyadh Metro Assistant that handles multiple data types
    with appropriate chunking and metadata for each collection.
    """
    
    def __init__(self, persist_path='./chromadb'):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def index_all_data(self):
        """
        Index all data files in the repository/data directory
        """
        logger.info("Starting comprehensive data indexing for Riyadh Metro Assistant")
        
        # Index each data type with appropriate processing
        self._index_pharmacy_products()
        self._index_kafd_rules()
        self._index_football_data()
        self._index_alnasr_text()
        self._index_ticketing_data()
        self._index_amenities_data()
        
        logger.info("Completed indexing all data files")
    
    def _index_pharmacy_products(self):
        """Index pharmacy products with detailed product information"""
        logger.info("Indexing pharmacy products...")
        
        collection_name = "pharmacy_products"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/pharmacy_products.json"
        if not os.path.exists(data_path):
            logger.warning(f"Pharmacy products file not found: {data_path}")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for product in products:
            # Create a comprehensive document string
            doc_content = f"""
            Product: {product.get('BrandName', 'Unknown')}
            Generic Name: {product.get('GenericName', 'N/A')}
            Description: {product.get('Description', 'N/A')}
            Common Uses: {product.get('CommonUses', 'N/A')}
            Manufacturer: {product.get('Manufacturer', 'N/A')}
            Status: {product.get('Rx_OTC_Status', 'N/A')}
            Package Size: {product.get('PackageSize', 'N/A')}
            Price: ${product.get('UnitPrice_USD', 'N/A')}
            """
            
            documents.append(doc_content.strip())
            metadatas.append({
                "type": "pharmacy_product",
                "brand_name": product.get('BrandName', ''),
                "generic_name": product.get('GenericName', ''),
                "manufacturer": product.get('Manufacturer', ''),
                "status": product.get('Rx_OTC_Status', ''),
                "price": product.get('UnitPrice_USD', 0)
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed {len(products)} pharmacy products")
    
    def _index_kafd_rules(self):
        """Index KAFD rules and metro information"""
        logger.info("Indexing KAFD rules and metro information...")
        
        collection_name = "metro_rules"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/KAFD_rules.json"
        if not os.path.exists(data_path):
            logger.warning(f"KAFD rules file not found: {data_path}")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for section in rules_data:
            section_name = section.get('section', 'Unknown')
            
            # Process each section based on its type
            if section_name == "Train Rules and Safety":
                # Handle safety and rules information
                safety_info = section.get('onboard_and_station_safety', {})
                passenger_guidelines = section.get('passenger_guidelines', {})
                
                doc_content = f"""
                Section: {section_name}
                Security Measures: {', '.join(safety_info.get('security_measures', []))}
                Operations: {', '.join(safety_info.get('operations', []))}
                Noise Policy: {passenger_guidelines.get('noise_policy', 'N/A')}
                Safety Rules: {', '.join(passenger_guidelines.get('safety_rules', []))}
                Platform Guidance: {passenger_guidelines.get('platform_guidance', 'N/A')}
                """
                
                documents.append(doc_content.strip())
                metadatas.append({
                    "type": "metro_rules",
                    "section": section_name,
                    "category": "safety_and_rules"
                })
                ids.append(str(uuid.uuid4()))
                
            elif section_name == "Etiquette and Train Classes":
                # Handle train classes and etiquette
                train_classes = section.get('train_classes', [])
                features = section.get('features', [])
                
                for train_class in train_classes:
                    class_name = train_class.get('class', 'Unknown')
                    descriptions = train_class.get('description', [])
                    
                    doc_content = f"""
                    Train Class: {class_name}
                    Descriptions: {', '.join(descriptions)}
                    Features: {', '.join(features)}
                    Cultural Context: {section.get('cultural_context', 'N/A')}
                    """
                    
                    documents.append(doc_content.strip())
                    metadatas.append({
                        "type": "metro_rules",
                        "section": section_name,
                        "category": "train_classes",
                        "class_name": class_name
                    })
                    ids.append(str(uuid.uuid4()))
                    
            elif section_name == "Accessibility Features":
                # Handle accessibility information
                station_design = section.get('station_design', [])
                train_features = section.get('train_features', [])
                facilities = section.get('facilities', [])
                
                doc_content = f"""
                Section: {section_name}
                Station Design: {', '.join(station_design)}
                Train Features: {', '.join(train_features)}
                Facilities: {', '.join(facilities)}
                Service Animals: {section.get('service_animals', 'N/A')}
                Overview: {section.get('overview', 'N/A')}
                """
                
                documents.append(doc_content.strip())
                metadatas.append({
                    "type": "metro_rules",
                    "section": section_name,
                    "category": "accessibility"
                })
                ids.append(str(uuid.uuid4()))
                
            elif section.get('location') == "KAFD Nearby Services":
                # Handle KAFD amenities
                features = section.get('features', [])
                
                for feature in features:
                    feature_type = feature.get('type', 'Unknown')
                    details = feature.get('details', 'N/A')
                    examples = feature.get('examples', [])
                    range_info = feature.get('range', 'N/A')
                    
                    doc_content = f"""
                    Location: KAFD Nearby Services
                    Feature Type: {feature_type}
                    Details: {details}
                    Range: {range_info}
                    Examples: {', '.join(examples) if examples else 'N/A'}
                    Summary: {section.get('summary', 'N/A')}
                    """
                    
                    documents.append(doc_content.strip())
                    metadatas.append({
                        "type": "amenities",
                        "location": "KAFD",
                        "feature_type": feature_type,
                        "category": "nearby_services"
                    })
                    ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed KAFD rules and metro information")
    
    def _index_football_data(self):
        """Index football team and stadium information"""
        logger.info("Indexing football data...")
        
        collection_name = "sports_venues"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/football_data.json"
        if not os.path.exists(data_path):
            logger.warning(f"Football data file not found: {data_path}")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            football_data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        teams = football_data.get('football teams', [])
        for team in teams:
            team_name = team.get('team', 'Unknown')
            stadium = team.get('stadium', 'N/A')
            capacity = team.get('capacity', 'N/A')
            nearest_station = team.get('nearest_metro_station', team.get('closest_station', 'N/A'))
            metro_line = team.get('metro_line', 'N/A')
            distance = team.get('distance_from_nearest_metro_station', 'N/A')
            location = team.get('location', 'N/A')
            
            doc_content = f"""
            Team: {team_name}
            Stadium: {stadium}
            Location: {location}
            Capacity: {capacity}
            Nearest Metro Station: {nearest_station}
            Metro Line: {metro_line}
            Distance from Station: {distance}
            """
            
            documents.append(doc_content.strip())
            metadatas.append({
                "type": "sports_venue",
                "team": team_name,
                "stadium": stadium,
                "metro_line": metro_line,
                "category": "football"
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed {len(teams)} football teams and venues")
    
    def _index_alnasr_text(self):
        """Index Al-Nassr text content with chunking"""
        logger.info("Indexing Al-Nassr text content...")
        
        collection_name = "sports_content"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/alnasr.txt"
        if not os.path.exists(data_path):
            logger.warning(f"Al-Nassr text file not found: {data_path}")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "type": "sports_content",
                "team": "Al-Nassr",
                "content_type": "season_info",
                "chunk_index": i
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed Al-Nassr content in {len(chunks)} chunks")
    
    def _index_ticketing_data(self):
        """Index ticketing information"""
        logger.info("Indexing ticketing data...")
        
        collection_name = "ticketing"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/ticketing_data.json"
        if not os.path.exists(data_path):
            logger.warning(f"Ticketing data file not found: {data_path}")
            return
        
        # Check if file is empty
        if os.path.getsize(data_path) == 0:
            logger.warning("Ticketing data file is empty")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            ticketing_data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        # Process ticketing data (structure may vary)
        if isinstance(ticketing_data, list):
            for item in ticketing_data:
                doc_content = str(item)
                documents.append(doc_content)
                metadatas.append({
                    "type": "ticketing",
                    "category": "fare_info"
                })
                ids.append(str(uuid.uuid4()))
        else:
            # Handle as single object
            doc_content = str(ticketing_data)
            documents.append(doc_content)
            metadatas.append({
                "type": "ticketing",
                "category": "fare_info"
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed ticketing data")
    
    def _index_amenities_data(self):
        """Index amenities data"""
        logger.info("Indexing amenities data...")
        
        collection_name = "amenities"
        collection = self._get_or_create_collection(collection_name)
        
        data_path = "./data/amenities.json"
        if not os.path.exists(data_path):
            logger.warning(f"Amenities data file not found: {data_path}")
            return
        
        # Check if file is empty
        if os.path.getsize(data_path) == 0:
            logger.warning("Amenities data file is empty")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            amenities_data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        # Process amenities data (structure may vary)
        if isinstance(amenities_data, list):
            for item in amenities_data:
                doc_content = str(item)
                documents.append(doc_content)
                metadatas.append({
                    "type": "amenities",
                    "category": "station_amenities"
                })
                ids.append(str(uuid.uuid4()))
        else:
            # Handle as single object
            doc_content = str(amenities_data)
            documents.append(doc_content)
            metadatas.append({
                "type": "amenities",
                "category": "station_amenities"
            })
            ids.append(str(uuid.uuid4()))
        
        self._add_to_collection(collection, ids, documents, metadatas)
        logger.info(f"Indexed amenities data")
    
    def _get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"Using collection: {collection_name}")
            return collection
        except Exception as e:
            raise Exception(f"Error creating or getting collection '{collection_name}': {e}")
    
    def _add_to_collection(self, collection, ids: List[str], documents: List[str], metadatas: List[Dict]):
        """Add documents to a ChromaDB collection with embeddings"""
        if not documents:
            logger.warning("No documents to add to collection")
            return
        
        try:
            # Generate embeddings for all documents
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_model.embed_documents(documents)
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(ids)} items to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            raise

def push_to_chroma(json_data_path: str = "./data/pharmacy_products.json", persist_path: str = './chromadb'):
    """
    Legacy function for backward compatibility - now delegates to the new indexer
    """
    logger.info(f"Using legacy push_to_chroma function for {json_data_path}")
    
    indexer = MetroDataIndexer(persist_path)
    
    # Determine which indexing method to use based on the file path
    if "pharmacy_products" in json_data_path:
        indexer._index_pharmacy_products()
    elif "KAFD_rules" in json_data_path:
        indexer._index_kafd_rules()
    elif "football_data" in json_data_path:
        indexer._index_football_data()
    elif "alnasr" in json_data_path:
        indexer._index_alnasr_text()
    elif "ticketing_data" in json_data_path:
        indexer._index_ticketing_data()
    elif "amenities" in json_data_path:
        indexer._index_amenities_data()
    else:
        # Default to pharmacy products for backward compatibility
        indexer._index_pharmacy_products()

def index_all_metro_data(persist_path: str = './chromadb'):
    """
    Index all metro-related data files
    """
    indexer = MetroDataIndexer(persist_path)
    indexer.index_all_data()

