import re
import string
import nltk
from typing import List, Dict, Tuple, Any
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import jieba

class TextProcessor:
    """针对中英文文本进行优化"""
    
    def __init__(self):
        """Initialize text processor"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK resources: {e}")

        self.stopwords = self._load_stopwords()

        self.lemmatizer = WordNetLemmatizer()

        jieba.initialize()
        
    def _load_stopwords(self) -> set:
        """Load stopwords for filtering"""
        try:
            english_stopwords = set(stopwords.words('english'))

            additional_stopwords = {
                'would', 'could', 'should', 'might', 'may', 'can', 'cannot',
                'must', 'shall', 'will', 'have', 'has', 'had', 'do', 'does',
                'did', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'this', 'that', 'these', 'those', 'there', 'here'
            }
            
            return english_stopwords.union(additional_stopwords)
        except:
            return {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
                'will', 'just', 'don', 'should', 'now'
            }
        
    def preprocess(self, text: str) -> str:
        """
        Preprocess input text
        
        Parameters:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize contractions
        text = self._expand_contractions(text)
        
        # Standardize punctuation
        text = self._normalize_punctuation(text)
        
        return text
        
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions"""
        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", 
            "can't've": "cannot have", "'cause": "because", "could've": "could have", 
            "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", 
            "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
            "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
            "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
            "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
            "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
            "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
            "i'll've": "i will have", "i'm": "i am", "i've": "i have", 
            "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
            "it'll": "it will", "it'll've": "it will have", "it's": "it is", 
            "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
            "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", 
            "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
            "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", 
            "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
            "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
            "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
            "she's": "she is", "should've": "should have", "shouldn't": "should not", 
            "shouldn't've": "should not have", "so've": "so have", "so's": "so is", 
            "that'd": "that would", "that'd've": "that would have", "that's": "that is", 
            "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
            "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
            "they'll've": "they will have", "they're": "they are", "they've": "they have", 
            "to've": "to have", "wasn't": "was not", "we'd": "we would", 
            "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
            "we're": "we are", "we've": "we have", "weren't": "were not", 
            "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
            "what's": "what is", "what've": "what have", "when's": "when is", 
            "when've": "when have", "where'd": "where did", "where's": "where is", 
            "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
            "who's": "who is", "who've": "who have", "why's": "why is", 
            "why've": "why have", "will've": "will have", "won't": "will not", 
            "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
            "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", 
            "y'all'd've": "you all would have", "y'all're": "you all are", 
            "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", 
            "you'll": "you will", "you'll've": "you will have", "you're": "you are", 
            "you've": "you have"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text
        
    def _normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation"""
        text = re.sub(r'\s+', ' ', text)

        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
        
        return text
        
    def extract_claim(self, text: str) -> str:
        """
        Extract core claim from input text
        
        Parameters:
            text: Input text
            
        Returns:
            Extracted claim
        """
        # Preprocess text
        processed_text = self.preprocess(text)
        
        # Remove question prefixes
        claim = self._remove_question_prefixes(processed_text)
        
        # Standardize question format
        claim = self._standardize_question(claim)
        
        return claim
        
    def _remove_question_prefixes(self, text: str) -> str:
        """Remove common question prefixes"""
        prefixes = [
            r'^(please )?(tell|inform) me\s',
            r'^(please )?(let me know|explain)\s',
            r'^(can|could) you (tell|inform) me\s',
            r'^(can|could) you (let me know|explain)\s',
            r'^I (want|need|would like) to (know|understand|learn)\s',
            r'^I\'m (curious|wondering|asking) (about|if|whether)\s',
            r'^(Do you know|Is it true|Is it correct) (that|if|whether)\s',
            r'^(verify|confirm) (that|if|whether)\s',
            r'^(I\'m trying to|help me) (determine|verify|confirm)\s'
        ]
        
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
            
        return text
        
    def _standardize_question(self, text: str) -> str:
        """Standardize question format"""
        if text and not text.endswith('?'):
            question_patterns = [
                r'\b(who|what|where|when|why|how|which|whose)\b',
                r'\b(is|are|was|were|do|does|did|have|has|had|can|could|will|would|should|must)\b.*\b',
                r'\b(verify|confirm|check)\b'
            ]
            
            for pattern in question_patterns:
                if re.search(pattern, text, re.IGNORECASE) and not text.endswith('.'):
                    text += '?'
                    break
                    
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and filter stopwords
        
        Parameters:
            text: Input text
            
        Returns:
            List of tokens after filtering
        """
        if self._is_english(text):
            tokens = word_tokenize(text.lower())

            filtered_tokens = [
                token for token in tokens 
                if token not in self.stopwords and token not in string.punctuation
            ]

            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            return lemmatized_tokens
        else:
            # 中文优化
            tokens = jieba.lcut(text)

            filtered_tokens = [
                token for token in tokens 
                if token not in self.stopwords and token not in string.punctuation
            ]
            
            return filtered_tokens
            
    def _is_english(self, text: str) -> bool:
        """Check if text is primarily English"""
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / max(len(text), 1) > 0.6
        
    def extract_keywords(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Extract keywords from text
        
        Parameters:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List of keyword tuples (word, weight)
        """
        # For English text, use TF-IDF approach
        if self._is_english(text):
            tokens = self.tokenize(text)
            
            if not tokens:
                return []

            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            keywords = sorted(tf.items(), key=lambda x: x[1], reverse=True)

            return [(word, float(freq) / len(tokens)) for word, freq in keywords[:top_n]]
        else:
            try:
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(text, topK=top_n, withWeight=True)
                return keywords
            except:
                tokens = jieba.lcut(text)
                tf = {}
                for token in tokens:
                    if token not in self.stopwords and token not in string.punctuation:
                        tf[token] = tf.get(token, 0) + 1
                
                keywords = sorted(tf.items(), key=lambda x: x[1], reverse=True)
                return [(word, float(freq) / len(tokens)) for word, freq in keywords[:top_n]]
        
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with variations
        
        Parameters:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        expanded_queries = [query]
        
        # Extract keywords
        keywords = self.extract_keywords(query)
        
        if keywords:
            keyword_only_query = ' '.join([k for k, _ in keywords])
            expanded_queries.append(keyword_only_query)

            if len(keywords) > 1:
                reversed_keywords = ' '.join([k for k, _ in reversed(keywords)])
                expanded_queries.append(reversed_keywords)

        is_english = self._is_english(query)
        if is_english:
            if query.endswith('?'):
                statement = query[:-1].strip()

                statement = re.sub(r'^(is|are|was|were|do|does|did|have|has|had|can|could|will|would|should|must)\s+(.+?)\s+(.+)$', 
                                   r'\2 \1 \3', statement, flags=re.IGNORECASE)
                
                expanded_queries.append(statement)
            else:
                if not any(w in query.lower() for w in ['who', 'what', 'where', 'when', 'why', 'how']):
                    question = f"Is it true that {query.lower()}?"
                    expanded_queries.append(question)
            
            # Add negation form
            negation_patterns = [
                (r'\bis\b', 'is not'),
                (r'\bare\b', 'are not'),
                (r'\bwas\b', 'was not'),
                (r'\bwere\b', 'were not'),
                (r'\bdo\b', 'do not'),
                (r'\bdoes\b', 'does not'),
                (r'\bdid\b', 'did not'),
                (r'\bhave\b', 'have not'),
                (r'\bhas\b', 'has not'),
                (r'\bhad\b', 'had not'),
                (r'\bcan\b', 'cannot'),
                (r'\bcould\b', 'could not'),
                (r'\bwill\b', 'will not'),
                (r'\bwould\b', 'would not'),
                (r'\bshould\b', 'should not'),
                (r'\bmust\b', 'must not')
            ]

            if not any(neg[1] in query.lower() for neg in negation_patterns):
                for pattern, replacement in negation_patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        negated_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                        expanded_queries.append(negated_query)
                        break
        
        return list(set(expanded_queries))
        
    def process_for_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Prepare query for retrieval
        
        Parameters:
            query: Original query text
            
        Returns:
            Dictionary with processed query information
        """
        processed_query = self.preprocess(query)
        claim = self.extract_claim(processed_query)

        tokens = self.tokenize(claim)

        keywords = self.extract_keywords(claim)

        expanded_queries = self.expand_query(claim)
        
        return {
            'original_query': query,
            'processed_query': processed_query,
            'claim': claim,
            'tokens': tokens,
            'keywords': keywords,
            'expanded_queries': expanded_queries,
            'is_english': self._is_english(query)
        }