import argparse
import logging
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchRequest
from qdrant_client.http.exceptions import UnexpectedResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "document_collection"
VECTOR_DIM = 1024  # Dimension of vectors in your collection

class QdrantQueryTool:
    """A tool for querying Qdrant without embedding or document loading"""
    
    def __init__(self, url="http://localhost:6333", api_key=None, collection_name=COLLECTION_NAME):
        """Initialize connection to Qdrant"""
        self.collection_name = collection_name
        
        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {url}")
        try:
            if api_key:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                self.client = QdrantClient(url=url)
                
            # Test connection
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.error(f"Collection '{self.collection_name}' not found in Qdrant. Available collections: {collection_names}")
                raise ValueError(f"Collection '{self.collection_name}' not found")
            
            logger.info(f"Successfully connected to Qdrant. Collection '{self.collection_name}' is available.")
            
            # Get collection info to verify dimension
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            actual_dim = collection_info.config.params.vectors.size
            logger.info(f"Collection vector dimension: {actual_dim}")
            
            # Set the correct vector dimension
            global VECTOR_DIM
            VECTOR_DIM = actual_dim
            logger.info(f"Updated VECTOR_DIM to match collection: {VECTOR_DIM}")
            
            # Check collection points count
            points_count = collection_info.points_count
            logger.info(f"Collection contains {points_count} points")
            
            # Examine payload schema if available (newer Qdrant versions)
            if hasattr(collection_info, 'payload_schema'):
                logger.info(f"Payload schema: {collection_info.payload_schema}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def keyword_search(self, keyword, limit=10):
        """Perform a keyword search on text field"""
        if not keyword:
            logger.error("No keyword provided for search")
            return []
        
        logger.info(f"Performing keyword search for: {keyword}")
        
        try:
            # Attempt a more general search approach - get all points and filter locally
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=100  # Retrieve a larger batch to filter through
            )[0]
            
            logger.info(f"Retrieved {len(all_points)} total points to search through")
            
            # Filter locally based on keyword
            results = []
            for idx, point in enumerate(all_points):
                # Get text from the payload (could be in different field names)
                text = ""
                if "text" in point.payload:
                    text = point.payload["text"]
                elif "content" in point.payload:
                    text = point.payload["content"]
                
                # Get metadata from the payload
                metadata = {}
                if "metadata" in point.payload:
                    metadata = point.payload["metadata"]
                
                # Check payload for the keyword (case insensitive)
                found = False
                if keyword.lower() in text.lower():
                    found = True
                
                # Also search through metadata values
                if not found:
                    for key, value in point.payload.items():
                        if isinstance(value, str) and keyword.lower() in value.lower():
                            found = True
                            break
                
                if found:
                    # Print detailed debug for the first few matches
                    if len(results) < 3:
                        logger.debug(f"Match found in point {point.id}:")
                        logger.debug(f"Point payload keys: {list(point.payload.keys())}")
                        # Log a sample of matching text (truncated)
                        match_text = text[:100] + "..." if len(text) > 100 else text
                        logger.debug(f"Matching content: {match_text}")
                    
                    results.append({
                        "id": len(results) + 1,
                        "point_id": point.id,
                        "text": text,
                        "metadata": metadata,
                        "payload": point.payload  # Include full payload for inspection
                    })
                    
                    if len(results) >= limit:
                        break
            
            logger.info(f"Found {len(results)} results for keyword: {keyword}")
            return results
            
        except Exception as e:
            logger.error(f"Error during keyword search: {str(e)}")
            logger.exception("Full exception details:")
            return []
    
    def vector_search(self, vector, limit=10):
        """Perform a vector similarity search using a provided vector"""
        if not vector or len(vector) != VECTOR_DIM:
            logger.error(f"Invalid vector provided. Expected dimension: {VECTOR_DIM}")
            return []
        
        logger.info("Performing vector similarity search")
        
        try:
            # Search with the vector
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit
            )
            
            # Format results
            results = []
            for idx, result in enumerate(search_result):
                results.append({
                    "id": idx + 1,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.info(f"Found {len(results)} results for vector search")
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            return []
    
    def hybrid_search(self, vector, keyword=None, limit=10):
        """Perform a hybrid search using both vector and keyword if provided"""
        if not vector or len(vector) != VECTOR_DIM:
            logger.error(f"Invalid vector provided. Expected dimension: {VECTOR_DIM}")
            return []
        
        logger.info(f"Performing hybrid search with keyword: {keyword}")
        
        try:
            # Create filter if keyword is provided
            search_filter = None
            if keyword:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchValue(value=keyword)
                        )
                    ]
                )
            
            # Search with vector and optional filter
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=search_filter,
                limit=limit
            )
            
            # Format results
            results = []
            for idx, result in enumerate(search_result):
                results.append({
                    "id": idx + 1,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.info(f"Found {len(results)} results for hybrid search")
            return results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []
    
    def metadata_search(self, field, value, limit=10):
        """Search based on metadata field value"""
        if not field or not value:
            logger.error("Both field and value must be provided for metadata search")
            return []
        
        logger.info(f"Performing metadata search for {field}={value}")
        
        try:
            # Create a filter for the metadata field
            metadata_filter = Filter(
                must=[
                    FieldCondition(
                        key=f"metadata.{field}",
                        match=MatchValue(value=value)
                    )
                ]
            )
            
            # Use scroll method with correct parameter name "scroll_filter" instead of "filter"
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=metadata_filter,  # Use scroll_filter instead of filter
                limit=limit
            )
            
            # Format results
            results = []
            for idx, result in enumerate(scroll_result[0]):  # scroll returns (points, next_page_offset)
                results.append({
                    "id": idx + 1,
                    "point_id": result.id,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.info(f"Found {len(results)} results for metadata search")
            return results
            
        except Exception as e:
            logger.error(f"Error during metadata search: {str(e)}")
            return []
    
    def get_document_by_id(self, point_id):
        """Retrieve a document by its ID"""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            
            if not points:
                logger.warning(f"No document found with ID: {point_id}")
                return None
            
            point = points[0]
            result = {
                "id": point.id,
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving document with ID {point_id}: {str(e)}")
            return None
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            
            # Convert to a serializable dict
            info_dict = {
                "name": info.name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance
            }
            
            return info_dict
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return None
    
    def list_available_collections(self):
        """List all available collections in Qdrant"""
        try:
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            return collection_names
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

def main():
    """Main function to run the Qdrant query tool"""
    parser = argparse.ArgumentParser(description="Qdrant Query Tool")
    parser.add_argument("--url", type=str, default="http://localhost:6333", help="Qdrant server URL")
    parser.add_argument("--api_key", type=str, help="Qdrant API key (if needed)")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Collection name")
    parser.add_argument("--keyword", type=str, help="Keyword to search for")
    parser.add_argument("--vector_file", type=str, help="JSON file containing a vector to search with")
    parser.add_argument("--metadata_field", type=str, help="Metadata field to search on")
    parser.add_argument("--metadata_value", type=str, help="Value to match in metadata field")
    parser.add_argument("--point_id", type=int, help="Point ID to retrieve")
    parser.add_argument("--info", action="store_true", help="Get collection info")
    parser.add_argument("--list", action="store_true", help="List available collections")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize the query tool
        query_tool = QdrantQueryTool(
            url=args.url,
            api_key=args.api_key,
            collection_name=args.collection
        )
        
        results = None
        
        # List collections if requested
        if args.list:
            collections = query_tool.list_available_collections()
            results = {"collections": collections}
            logger.info(f"Available collections: {collections}")
        
        # Get collection info if requested
        elif args.info:
            info = query_tool.get_collection_info()
            results = {"collection_info": info}
            logger.info(f"Collection info: {json.dumps(info, indent=2)}")
        
        # Get document by ID if requested
        elif args.point_id is not None:
            document = query_tool.get_document_by_id(args.point_id)
            results = {"document": document}
            if document:
                logger.info(f"Retrieved document with ID {args.point_id}")
            else:
                logger.error(f"Failed to retrieve document with ID {args.point_id}")
        
        # Metadata search if both field and value are provided
        elif args.metadata_field and args.metadata_value:
            metadata_results = query_tool.metadata_search(
                args.metadata_field,
                args.metadata_value,
                limit=args.limit
            )
            results = {"results": metadata_results}
            logger.info(f"Found {len(metadata_results)} results for metadata search")
        
        # Keyword search if only keyword is provided
        elif args.keyword and not args.vector_file:
            keyword_results = query_tool.keyword_search(args.keyword, limit=args.limit)
            results = {"results": keyword_results}
            logger.info(f"Found {len(keyword_results)} results for keyword search")
        
        # Vector search if only vector file is provided
        elif args.vector_file and not args.keyword:
            # Load vector from file
            with open(args.vector_file, 'r') as f:
                vector_data = json.load(f)
                vector = vector_data.get("vector")
                
                if not vector:
                    logger.error("Vector file does not contain a 'vector' field")
                    return
            
            vector_results = query_tool.vector_search(vector, limit=args.limit)
            results = {"results": vector_results}
            logger.info(f"Found {len(vector_results)} results for vector search")
        
        # Hybrid search if both keyword and vector file are provided
        elif args.keyword and args.vector_file:
            # Load vector from file
            with open(args.vector_file, 'r') as f:
                vector_data = json.load(f)
                vector = vector_data.get("vector")
                
                if not vector:
                    logger.error("Vector file does not contain a 'vector' field")
                    return
            
            hybrid_results = query_tool.hybrid_search(vector, args.keyword, limit=args.limit)
            results = {"results": hybrid_results}
            logger.info(f"Found {len(hybrid_results)} results for hybrid search")
        
        else:
            logger.error("No search criteria provided. Use --keyword, --vector_file, --metadata_field and --metadata_value, --point_id, --info, or --list")
            return
        
        # Output results to file if requested
        if args.output and results:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Print results to console
        if results:
            print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    main()
    
    # Initialize the query tool
    query_tool = QdrantQueryTool(
        url="http://localhost:6333",
        collection_name="document_collection"
    )
    
    # Perform a keyword search
    results = query_tool.keyword_search("name")
    
    # Print the results
    for result in results:
        print(f"Document: {result['text'][:100]}...")
        print(f"Source: {result['metadata'].get('source', 'Unknown')}")
        print("-" * 50)
