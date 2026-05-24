"""
Updated augmentation_pipeline.py with proper MeSH support
Supports both desc (descriptors) and supp (supplementary concepts)
"""

import xml.etree.ElementTree as ET
import gzip
from pathlib import Path
from typing import List, Dict, Optional, Set
import re
import random
from collections import defaultdict


class MeSHSynonymExtractor:
    """
    Extract synonyms from MeSH database
    Supports both Descriptors (desc2025.xml) and Supplementary Concepts (supp2025.xml)
    """
    
    def __init__(self, mesh_desc_path: str, mesh_supp_path: str = None):
        """
        Args:
            mesh_desc_path: Path to desc2025.xml or desc2025.gz (required)
            mesh_supp_path: Path to supp2025.xml or supp2025.gz (optional but recommended)
        """
        self.mesh_desc_path = Path(mesh_desc_path)
        self.mesh_supp_path = Path(mesh_supp_path) if mesh_supp_path else None
        
        self.concept_to_synonyms = {}
        self.term_to_concept = {}
        
        print("=" * 60)
        print("Loading MeSH Database")
        print("=" * 60)
        self._load_mesh()
        print(f"✓ Total concepts loaded: {len(self.concept_to_synonyms)}")
        print(f"✓ Total terms indexed: {len(self.term_to_concept)}")
        print("=" * 60)
    
    def _load_xml_file(self, file_path: Path):
        """Helper to load XML file (handles both .xml and .gz)"""
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return ET.parse(f)
        else:
            return ET.parse(file_path)
    
    def _load_mesh(self):
        """Parse MeSH XML files and build synonym mappings"""
        
        # Load Descriptors (main concepts)
        print(f"Loading Descriptors from: {self.mesh_desc_path}")
        desc_count = self._load_descriptors()
        print(f"  ✓ Loaded {desc_count} descriptors")
        
        # Load Supplementary Concepts (additional terms)
        if self.mesh_supp_path and self.mesh_supp_path.exists():
            print(f"Loading Supplementary Concepts from: {self.mesh_supp_path}")
            supp_count = self._load_supplementary()
            print(f"  ✓ Loaded {supp_count} supplementary concepts")
        else:
            if self.mesh_supp_path:
                print(f"  ⚠ Supplementary file not found: {self.mesh_supp_path}")
            print("  ℹ Running without supplementary concepts")
    
    def _load_descriptors(self) -> int:
        """Load MeSH Descriptors"""
        tree = self._load_xml_file(self.mesh_desc_path)
        root = tree.getroot()
        
        count = 0
        for descriptor in root.findall('.//DescriptorRecord'):
            # Get descriptor UI (unique ID)
            desc_ui_elem = descriptor.find('DescriptorUI')
            if desc_ui_elem is None:
                continue
            desc_ui = desc_ui_elem.text
            
            # Get descriptor name
            desc_name_elem = descriptor.find('.//DescriptorName/String')
            if desc_name_elem is None:
                continue
            desc_name = desc_name_elem.text
            
            # Collect all synonyms
            synonyms = {desc_name.lower()}
            
            # Add terms from ConceptList
            for concept in descriptor.findall('.//Concept'):
                for term in concept.findall('.//Term'):
                    term_string = term.find('String')
                    if term_string is not None:
                        synonyms.add(term_string.text.lower())
            
            # Store mappings
            self.concept_to_synonyms[desc_ui] = list(synonyms)
            for syn in synonyms:
                self.term_to_concept[syn] = desc_ui
            
            count += 1
        
        return count
    
    def _load_supplementary(self) -> int:
        """Load MeSH Supplementary Concepts"""
        tree = self._load_xml_file(self.mesh_supp_path)
        root = tree.getroot()
        
        count = 0
        for supp_record in root.findall('.//SupplementalRecord'):
            # Get supplementary concept UI
            supp_ui_elem = supp_record.find('SupplementalRecordUI')
            if supp_ui_elem is None:
                continue
            supp_ui = supp_ui_elem.text
            
            # Get supplementary concept name
            supp_name_elem = supp_record.find('.//SupplementalRecordName/String')
            if supp_name_elem is None:
                continue
            supp_name = supp_name_elem.text
            
            # Collect synonyms
            synonyms = {supp_name.lower()}
            
            # Add terms from ConceptList
            for concept in supp_record.findall('.//Concept'):
                for term in concept.findall('.//Term'):
                    term_string = term.find('String')
                    if term_string is not None:
                        synonyms.add(term_string.text.lower())
            
            # Store mappings
            self.concept_to_synonyms[supp_ui] = list(synonyms)
            for syn in synonyms:
                # Don't override descriptor mappings with supplementary ones
                if syn not in self.term_to_concept:
                    self.term_to_concept[syn] = supp_ui
            
            count += 1
        
        return count
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        Get all synonyms for a given term
        
        Args:
            term: Medical term to find synonyms for
        
        Returns:
            List of synonyms (excluding the input term itself)
        """
        term_lower = term.lower()
        if term_lower in self.term_to_concept:
            concept_id = self.term_to_concept[term_lower]
            # Return all synonyms except the input term
            syns = [s for s in self.concept_to_synonyms[concept_id] 
                   if s != term_lower]
            return syns
        return []
    
    def replace_with_synonym(self, text: str, term: str, avoid_terms: Set[str] = None) -> str:
        """
        Replace a term with one of its synonyms
        
        Args:
            text: Text to modify
            term: Term to replace
            avoid_terms: Set of terms to avoid using as replacements
        
        Returns:
            Modified text with term replaced
        """
        synonyms = self.get_synonyms(term)
        if not synonyms:
            return text
        
        # Filter out terms to avoid
        if avoid_terms:
            synonyms = [s for s in synonyms if s not in avoid_terms]
        
        if not synonyms:
            return text
        
        # Use a random synonym
        replacement = random.choice(synonyms)
        
        # Case-preserving replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        
        def replace_match(match):
            original = match.group(0)
            # Preserve case
            if original.isupper():
                return replacement.upper()
            elif original[0].isupper():
                return replacement.capitalize()
            else:
                return replacement
        
        return pattern.sub(replace_match, text, count=1)
    def replace_with_synonym_detailed(self, text: str, term: str, avoid_terms: Set[str] = None):
        """
        替换术语并返回详细信息（用于QA任务答案追踪）
        
        Args:
            text: 原文本
            term: 要替换的术语
            avoid_terms: 要避免使用的术语集合
        
        Returns:
            (new_text, replacement_info)
            - new_text: 替换后的文本
            - replacement_info: dict 包含 {
                'original': 原术语,
                'replacement': 替换后的术语,
                'position': 替换位置,
                'old_length': 原术语长度,
                'new_length': 新术语长度
              } 或 None（如果没有替换）
        """
        synonyms = self.get_synonyms(term)
        if not synonyms:
            return text, None
        
        # Filter out terms to avoid
        if avoid_terms:
            synonyms = [s for s in synonyms if s not in avoid_terms]
        
        if not synonyms:
            return text, None
        
        # Use a random synonym
        replacement = random.choice(synonyms)
        
        # Find the first occurrence
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return text, None
        
        position = match.start()
        original_term = match.group(0)  # 保留原始大小写
        
        # Case-preserving replacement
        def replace_match(m):
            original = m.group(0)
            if original.isupper():
                return replacement.upper()
            elif original[0].isupper():
                return replacement.capitalize()
            else:
                return replacement
        
        new_text = pattern.sub(replace_match, text, count=1)
        
        # Calculate the actual replacement used (with case preserved)
        actual_replacement = new_text[position:position + len(replacement)]
        
        # Return detailed information
        info = {
            'original': original_term,  
            'replacement': actual_replacement,  
            'position': position,
            'old_length': len(original_term),
            'new_length': len(actual_replacement)
        }
        
        return new_text, info

"""
improved SynonymAugmentor 
1. augment() - standard synonym replacement (no target needed)
2. augment_with_target_terms() - replacement based on target terms
"""

import re
from typing import List, Dict, Set

class SynonymAugmentor:
    """Method 1: Replace medical terms with MeSH synonyms"""
    
    def __init__(self, mesh_extractor):
        self.mesh = mesh_extractor
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract potential medical terms from text
        Simple heuristic: look for capitalized words and noun phrases
        """
        # Extract capitalized words and phrases
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Also extract some common lowercase medical terms
        common_medical = re.findall(
            r'\b(?:hypertension|diabetes|pneumonia|infection|disease|syndrome|disorder)\b',
            text,
            re.IGNORECASE
        )
        words.extend(common_medical)
        
        # Filter to terms that exist in MeSH
        medical_terms = []
        for word in words:
            if self.mesh.get_synonyms(word):
                medical_terms.append(word)
        
        return list(set(medical_terms))  # Remove duplicates
    
    def augment(self, retrieved_sources: List[Dict], 
                max_replacements: int = 3) -> List[Dict]:
        """
        Standard synonym replacement augmentation (no target needed)
        
        Args:
            retrieved_sources: List of similar source samples
            max_replacements: Maximum number of terms to replace per sample
        
        Returns:
            List of augmented samples (synonym-replaced versions only)
        """
        augmented = []
        
        for source in retrieved_sources:
            text_field = 'text'
            if text_field not in source:
                continue
            
            text = source[text_field]
            medical_terms = self.extract_medical_terms(text)
            
            if not medical_terms:
                continue
            
            terms_to_replace = medical_terms[:max_replacements]
            
            augmented_text = text
            replaced_terms = []
            
            for term in terms_to_replace:
                new_text = self.mesh.replace_with_synonym(
                    augmented_text,
                    term,
                    avoid_terms=set(replaced_terms)
                )
                if new_text != augmented_text:
                    replaced_terms.append(term)
                    augmented_text = new_text
            
            if replaced_terms:
                aug_sample = source.copy()
                aug_sample[text_field] = augmented_text
                aug_sample['augmentation_type'] = 'synonym_replacement'
                aug_sample['replaced_terms'] = replaced_terms
                augmented.append(aug_sample)
        
        return augmented
    
    def augment_with_target_terms(self, target_sample: Dict, 
                                   retrieved_sources: List[Dict],
                                   max_replacements: int = 3,
                                   strategy: str = 'common') -> List[Dict]:
        augmented = []
        
        target_text = target_sample.get('text', '')
        target_terms = set(self.extract_medical_terms(target_text))
        
        if not target_terms:
            return self.augment(retrieved_sources, max_replacements)
        
        for source in retrieved_sources:
            text_field = 'text'
            if text_field not in source:
                continue
            
            text = source[text_field]
            source_terms = self.extract_medical_terms(text)
            
            if not source_terms:
                continue

            if strategy == 'common':
                terms_to_consider = [t for t in source_terms if t in target_terms]
            elif strategy == 'all':
                common_terms = [t for t in source_terms if t in target_terms]
                other_terms = [t for t in source_terms if t not in target_terms]
                terms_to_consider = common_terms + other_terms
            else:
                terms_to_consider = source_terms
            
            if not terms_to_consider:
                continue
            
           
            terms_to_replace = terms_to_consider[:max_replacements]
            
            augmented_text = text
            replaced_terms = []
            
            for term in terms_to_replace:
                new_text = self.mesh.replace_with_synonym(
                    augmented_text,
                    term,
                    avoid_terms=set(replaced_terms)
                )
                if new_text != augmented_text:
                    replaced_terms.append(term)
                    augmented_text = new_text
            
            if replaced_terms:
                aug_sample = source.copy()
                aug_sample[text_field] = augmented_text
                aug_sample['augmentation_type'] = 'synonym_replacement_target_aligned'
                aug_sample['replaced_terms'] = replaced_terms
                aug_sample['target_terms_used'] = list(target_terms)
                augmented.append(aug_sample)
        
        return augmented
    
    def augment_single(self, source: Dict, max_replacements: int = 3) -> Dict:
        
        results = self.augment([source], max_replacements)
        return results[0] if results else source
    
    def augment_single_with_target(self, target_sample: Dict, 
                                   source: Dict,
                                   max_replacements: int = 3,
                                   strategy: str = 'common') -> Dict:
        results = self.augment_with_target_terms(
            target_sample, 
            [source], 
            max_replacements,
            strategy
        )
        return results[0] if results else source
    
    

# Keep StyleTransferAugmentor for compatibility
# (This would be implemented separately with API calls)
class StyleTransferAugmentor:
    """Method 2: Style transfer - placeholder for compatibility"""
    pass


if __name__ == "__main__":
    # Test MeSH loading
    print("\nTesting MeSH Synonym Extractor...")
    
    desc_path = "/project/wliu9/Dataset/MESH/desc2025.xml"
    supp_path = "/project/wliu9/Dataset/MESH/supp2025.xml"
    
    # Try loading
    try:
        mesh = MeSHSynonymExtractor(desc_path, supp_path)
        
        # Test some terms
        test_terms = ["hypertension", "diabetes", "pneumonia", "aspirin"]
        
        print("\nTesting synonym extraction:")
        for term in test_terms:
            synonyms = mesh.get_synonyms(term)
            print(f"\n{term}:")
            print(f"  Found {len(synonyms)} synonyms")
            if synonyms:
                print(f"  Examples: {synonyms[:3]}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()



######### eurolex
# utils/eurvoc_extractor.py
# 
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set

def _normalize_term(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[-\s]+", "", s)
    return s.lower()

def _variants_for(term: str) -> Set[str]:
    raw = term.strip()
    t = raw.replace("–", "-").replace("—", "-")
    no_space = re.sub(r"\s+", "", t)
    hyphened = re.sub(r"\s+", "-", t)
    return {raw, t, no_space, hyphened}


# utils/eurvoc_extractor.py

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set


def _normalize_term(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[-\s]+", "", s)
    return s.lower()


def _variants_for(term: str) -> Set[str]:
    raw = term.strip()
    t = raw.replace("–", "-").replace("—", "-")
    no_space = re.sub(r"\s+", "", t)
    hyphened = re.sub(r"\s+", "-", t)
    return {raw, t, no_space, hyphened}


class EurVocSynonymExtractor:
    """
    load

    - mode="eurovoc_xlsx":   EuroVoc Excel
    - mode="gpt_jsonl":     GPT+EuroVoc  JSONL 
      JSONL example:
        {"term": "...", "synonyms": ["...", "..."]}

    """

    def __init__(
        self,
        path: str,
        sheet_name: str | int | None = 0,
        mode: str = "eurovoc_xlsx",
    ):
        self.path = Path(path)
        self.sheet_name = sheet_name
        self.mode = mode

 
        self.concept_to_terms: Dict[str, Set[str]] = {}
        
        self.concept_to_canonical: Dict[str, str] = {}
   
        self.termkey_to_concepts: Dict[str, Set[str]] = {}
       
        self.all_terms_for_match: Set[str] = set()

        self._load()

   

    def get_synonyms(self, term: str) -> List[str]:
        
        key = _normalize_term(term)
        if not key or key not in self.termkey_to_concepts:
            return []
        syns: Set[str] = set()
        for cid in self.termkey_to_concepts[key]:
            for t in self.concept_to_terms.get(cid, []):
                if _normalize_term(t) != key:
                    syns.add(t)
        return sorted(syns, key=lambda s: (-len(s), s.lower()))

    def get_canonical(self, term: str) -> str | None:
        
        key = _normalize_term(term)
        if not key or key not in self.termkey_to_concepts:
            return None
        
        cid = next(iter(self.termkey_to_concepts[key]))
        return self.concept_to_canonical.get(cid)

    def replace_with_synonym(
        self,
        text: str,
        term: str,
        avoid_terms: Set[str] | None = None,
    ) -> str:
      
        import re as _re

        avoid_terms = avoid_terms or set()
        canonical = self.get_canonical(term)
        if not canonical or canonical in avoid_terms:
            return text

        replacement = canonical

        pat = _re.compile(_re.escape(term), flags=_re.IGNORECASE)

        def _repl(m):
            orig = m.group(0)
            if orig.isupper():
                return replacement.upper()
            if orig[0].isupper():
                return replacement[:1].upper() + replacement[1:]
            return replacement

        return pat.sub(_repl, text, count=1)



    def _load(self):
        if self.mode == "eurovoc_xlsx":
            self._load_from_xlsx()
        elif self.mode == "gpt_jsonl":
            self._load_from_jsonl()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.all_terms_for_match = {t for t in self.all_terms_for_match if len(t) >= 2}
        self._pattern = self._build_pattern() 
    def _build_pattern(self):
        escaped = sorted(
            [re.escape(t) for t in self.all_terms_for_match],
            key=len, reverse=True
        )
        if not escaped:
            return re.compile(r"(?!x)x")
        return re.compile("|".join(rf"\b{e}\b" for e in escaped), flags=re.IGNORECASE)

    def find_terms_in_text(self, text: str) -> List[str]:
        hits = {}
        for m in self._pattern.finditer(text):
            span = m.group(0)
            key = span.lower()
            if key not in hits:
                hits[key] = span
        return sorted(hits.values(), key=lambda s: (-len(s), s.lower()))
    
    def _load_from_xlsx(self):

        df = pd.read_excel(self.path, sheet_name=self.sheet_name, dtype=str).fillna("")
        col_id = [c for c in df.columns if str(c).strip().lower() == "id"][0]
        col_terms = [c for c in df.columns if "term" in str(c).lower()][0]  # TERMS (PT-NPT)
        col_pt = [c for c in df.columns if str(c).strip().lower() == "pt"]
        col_pt = col_pt[0] if col_pt else None

        by_id: Dict[str, Set[str]] = {}
        for _, row in df.iterrows():
            cid = row[col_id].strip()
            term = row[col_terms].strip()
            pt = row[col_pt].strip() if col_pt else ""
            if not cid:
                continue
            by_id.setdefault(cid, set())
            if term:
                by_id[cid].update(_variants_for(term))
            if pt:
                by_id[cid].update(_variants_for(pt))

        for cid, terms in by_id.items():
            terms = {t for t in terms if t}
            if not terms:
                continue
            self.concept_to_terms[cid] = terms
           
            self.concept_to_canonical[cid] = next(iter(terms))

            for t in terms:
                key = _normalize_term(t)
                if not key:
                    continue
                self.termkey_to_concepts.setdefault(key, set()).add(cid)
                self.all_terms_for_match.add(t)

    def _load_from_jsonl(self):
      
        cid_counter = 0

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                term = (obj.get("term") or "").strip()
                syns = obj.get("synonyms") or []
                if not term:
                    continue

                cid_counter += 1
                cid = f"gpt_{cid_counter}"
                canonical = term

                terms_here: Set[str] = set()
            
                terms_here.update(_variants_for(term))
                
                for s in syns:
                    s = (s or "").strip()
                    if not s:
                        continue
                    terms_here.update(_variants_for(s))

                terms_here = {t for t in terms_here if t}
                if not terms_here:
                    continue

                self.concept_to_terms[cid] = terms_here
                self.concept_to_canonical[cid] = canonical

                for t in terms_here:
                    key = _normalize_term(t)
                    if not key:
                        continue
                    self.termkey_to_concepts.setdefault(key, set()).add(cid)
                    self.all_terms_for_match.add(t)

class EurVocSynonymAugmentor:
    def __init__(self, extractor):  
        self.lex = extractor

    def _extract_terms(self, text: str) -> List[str]:
        if hasattr(self.lex, "find_terms_in_text"):
            return self.lex.find_terms_in_text(text)
        return []
    
 
    def augment_single(self, source: dict, max_replacements: int = 3) -> dict:
    
        results = self.augment([source], max_replacements=max_replacements)
        return results[0] if results else source

    def augment(self, retrieved_sources: List[Dict], max_replacements: int = 3) -> List[Dict]:
        augmented = []
        for src in retrieved_sources:
            if 'text' not in src: 
                continue
            text = src['text']
            terms = self._extract_terms(text)
            if not terms:
                continue
            terms_to_replace = terms[:max_replacements]
            aug_text = text
            replaced = []
            for term in terms_to_replace:
                new_text = self.lex.replace_with_synonym(aug_text, term, avoid_terms=set(replaced))
                if new_text != aug_text:
                    replaced.append(term)
                    aug_text = new_text
            if replaced:
                item = src.copy()
                item['text'] = aug_text
                item['augmentation_type'] = 'synonym_replacement_eurovoc'
                item['replaced_terms'] = replaced
                augmented.append(item)
        return augmented


# ########   arxiv
# import json

from pathlib import Path
from typing import Dict, List, Set, Any
import re
import json


def _norm_key(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[-\s]+", "", s)
    return s.lower()


def _variants_for(term: str) -> Set[str]:
    raw = term.strip()
    t = raw.replace("–", "-").replace("—", "-")
    no_space = re.sub(r"\s+", "", t)
    hyphened = re.sub(r"\s+", "-", t)
    return {raw, t, no_space, hyphened}


class CSOSynonymExtractor:

    def __init__(self, lexicon_jsonl_path: str):
        self.lexicon_path = Path(lexicon_jsonl_path)

        self.concept_to_terms: Dict[str, Set[str]] = {}
        self.concept_to_canonical: Dict[str, str] = {}
        self.formkey_to_concepts: Dict[str, Set[str]] = {}
        self.all_forms_for_match: Set[str] = set()

        self._load_jsonl()

    def _load_jsonl(self):
        cid = 0
        with self.lexicon_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                term = (obj.get("term") or "").strip()
                syns = obj.get("synonyms") or []
                if not term:
                    continue

                cid += 1
                concept_id = f"cso_gpt_{cid}"
                canonical = term

                forms: Set[str] = set()
                forms.update(_variants_for(term))
                for s in syns:
                    s = (s or "").strip()
                    if not s:
                        continue
                    forms.update(_variants_for(s))

                forms = {x for x in forms if x}
                if not forms:
                    continue

                self.concept_to_terms[concept_id] = forms
                self.concept_to_canonical[concept_id] = canonical

                for form in forms:
                    k = _norm_key(form)
                    if not k:
                        continue
                    self.formkey_to_concepts.setdefault(k, set()).add(concept_id)
                    self.all_forms_for_match.add(form)

        self.all_forms_for_match = {t for t in self.all_forms_for_match if len(t) >= 2}
        self._pattern = self._build_pattern()  # ← 加这一行
        print("=" * 60)
        print(f"Loading ArXiv-CS JSONL lexicon from: {self.lexicon_path}")
        print(f"✓ Concepts loaded: {len(self.concept_to_terms)}")
        print(f"✓ Forms indexed: {len(self.all_forms_for_match)}")
        print("=" * 60)
    def _build_pattern(self):
        escaped = sorted(
            [re.escape(t) for t in self.all_forms_for_match],
            key=len, reverse=True
        )
        if not escaped:
            return re.compile(r"(?!x)x")
        return re.compile("|".join(rf"\b{e}\b" for e in escaped), flags=re.IGNORECASE)

    def find_terms_in_text(self, text: str) -> List[str]:
        hits = {}
        for m in self._pattern.finditer(text):
            span = m.group(0)
            key = span.lower()
            if key not in hits:
                hits[key] = span
        return sorted(hits.values(), key=lambda s: (-len(s), s.lower()))

    def get_canonical(self, form: str) -> str | None:
        key = _norm_key(form)
        if not key or key not in self.formkey_to_concepts:
            return None
        concept_id = next(iter(self.formkey_to_concepts[key]))
        return self.concept_to_canonical.get(concept_id)

    def replace_with_synonym(self, text: str, term: str, avoid_terms: Set[str] = None) -> str:
        """
        名字保留：但行为是 synonym→term（替换成 canonical term）
        """
        import re as _re

        avoid_terms = avoid_terms or set()
        canonical = self.get_canonical(term)
        if not canonical or canonical in avoid_terms:
            return text

        replacement = canonical
        pat = _re.compile(_re.escape(term), flags=_re.IGNORECASE)

        def _repl(m):
            orig = m.group(0)
            if orig.isupper():
                return replacement.upper()
            if orig and orig[0].isupper():
                return replacement[:1].upper() + replacement[1:]
            return replacement

        return pat.sub(_repl, text, count=1)

    def find_terms_in_text(self, text: str) -> List[str]:
        """
        找 text 中出现的任意 forms（包含 synonyms），按长度降序返回。
        """
        import re as _re

        hits: Set[str] = set()
        low = text.lower()
        for form in self.all_forms_for_match:
            fk = _norm_key(form)
            if fk and fk in _norm_key(low):
                if _re.compile(_re.escape(form), flags=_re.IGNORECASE).search(text):
                    hits.add(form)
        return sorted(hits, key=lambda s: (-len(s), s.lower()))


class CSOSynonymAugmentor:

    def __init__(self, lexicon_path: str):
        self.lexicon_path = Path(lexicon_path)
        self.lex = CSOSynonymExtractor(str(self.lexicon_path))

    def _extract_terms(self, text: str) -> List[str]:
        return self.lex.find_terms_in_text(text) if hasattr(self.lex, "find_terms_in_text") else []

    def augment(
        self,
        retrieved_sources: List[Dict[str, Any]],
        text_field: str = "text",
        max_replacements: int = 3,
    ) -> List[Dict[str, Any]]:
        augmented: List[Dict[str, Any]] = []

        for src in retrieved_sources:
            if text_field not in src:
                continue
            text = src[text_field]
            if not text:
                continue

            terms = self._extract_terms(text)
            if not terms:
                continue

            terms_to_replace = terms[:max_replacements]

            aug_text = text
            replaced = []
            replaced_lc: Set[str] = set()

            for t in terms_to_replace:
                new_text = self.lex.replace_with_synonym(aug_text, t, avoid_terms=replaced_lc)
                if new_text != aug_text:
                    aug_text = new_text
                    replaced.append(t)
                    replaced_lc.add(t.lower())

            if replaced:
                item = dict(src)
                item[text_field] = aug_text
                item["augmentation_type"] = "cso_synonym_to_term_canonicalization"
                item["replaced_terms"] = replaced
                augmented.append(item)

        return augmented

    def augment_single(
        self,
        source: Dict[str, Any],
        max_replacements: int = 3,
        text_field: str = "text",
    ) -> Dict[str, Any]:
        res = self.augment([source], text_field=text_field, max_replacements=max_replacements)
        return res[0] if res else dict(source)
