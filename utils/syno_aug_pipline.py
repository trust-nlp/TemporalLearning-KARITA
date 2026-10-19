"""
Knowledge-driven synonym augmentation used by KARITA (augmentation option1).

    MIMIC  : MeSHSynonymExtractor + MeSHSynonymAugmentor
             replace up to 3 MeSH terms with a random synonym of the same descriptor.
    EurLex : LexiconSynonymExtractor(mode="gpt_jsonl") + LexiconSynonymAugmentor
             GPT-4o-mini terms enriched with EuroVoc; rewrites any matched variant
             into its canonical term (synonym -> term normalisation).
    arXiv  : CSOWikidataSynonymAugmentor  (CSO lexicon, used for the paper results)
             or LexiconSynonymExtractor(mode="gpt_jsonl") with the GPT + CSO lexicon.

All augmentors expose  augment_single(source, max_replacements=3, text_field=...)
and return an augmented copy of `source` (or `source` unchanged when nothing
could be replaced).
"""

import gzip
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# =====================================================================
# MeSH (MIMIC-IV-Notes)
# =====================================================================

class MeSHSynonymExtractor:
    """
    Extract synonyms from MeSH.
    Supports Descriptors (desc2025.xml) and Supplementary Concepts (supp2025.xml).
    """

    def __init__(self, mesh_desc_path: str, mesh_supp_path: Optional[str] = None):
        self.mesh_desc_path = Path(mesh_desc_path)
        self.mesh_supp_path = Path(mesh_supp_path) if mesh_supp_path else None

        self.concept_to_synonyms: Dict[str, List[str]] = {}
        self.term_to_concept: Dict[str, str] = {}

        print("=" * 60)
        print("Loading MeSH Database")
        print("=" * 60)
        self._load_mesh()
        print(f"✓ Total concepts loaded: {len(self.concept_to_synonyms)}")
        print(f"✓ Total terms indexed: {len(self.term_to_concept)}")
        print("=" * 60)

    @staticmethod
    def _load_xml_file(file_path: Path):
        """Load an XML file (.xml or .gz)."""
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return ET.parse(f)
        return ET.parse(file_path)

    def _load_mesh(self):
        print(f"Loading Descriptors from: {self.mesh_desc_path}")
        desc_count = self._load_records(self.mesh_desc_path, 'DescriptorRecord',
                                        'DescriptorUI', 'DescriptorName', override=True)
        print(f"  ✓ Loaded {desc_count} descriptors")

        if self.mesh_supp_path and self.mesh_supp_path.exists():
            print(f"Loading Supplementary Concepts from: {self.mesh_supp_path}")
            # don't override descriptor mappings with supplementary ones
            supp_count = self._load_records(self.mesh_supp_path, 'SupplementalRecord',
                                            'SupplementalRecordUI', 'SupplementalRecordName',
                                            override=False)
            print(f"  ✓ Loaded {supp_count} supplementary concepts")
        else:
            if self.mesh_supp_path:
                print(f"  ⚠ Supplementary file not found: {self.mesh_supp_path}")
            print("  ℹ Running without supplementary concepts")

    def _load_records(self, path: Path, record_tag: str, ui_tag: str, name_tag: str,
                      override: bool) -> int:
        root = self._load_xml_file(path).getroot()
        count = 0
        for record in root.findall(f'.//{record_tag}'):
            ui_elem = record.find(ui_tag)
            name_elem = record.find(f'.//{name_tag}/String')
            if ui_elem is None or name_elem is None:
                continue
            ui = ui_elem.text

            synonyms = {name_elem.text.lower()}
            for concept in record.findall('.//Concept'):
                for term in concept.findall('.//Term'):
                    term_string = term.find('String')
                    if term_string is not None:
                        synonyms.add(term_string.text.lower())

            self.concept_to_synonyms[ui] = list(synonyms)
            for syn in synonyms:
                if override or syn not in self.term_to_concept:
                    self.term_to_concept[syn] = ui
            count += 1
        return count

    def get_synonyms(self, term: str) -> List[str]:
        """All synonyms of `term` (excluding the term itself)."""
        term_lower = term.lower()
        if term_lower in self.term_to_concept:
            concept_id = self.term_to_concept[term_lower]
            return [s for s in self.concept_to_synonyms[concept_id] if s != term_lower]
        return []

    def replace_with_synonym(self, text: str, term: str, avoid_terms: Set[str] = None) -> str:
        """Replace the first occurrence of `term` with a random synonym (case-preserving)."""
        synonyms = self.get_synonyms(term)
        if avoid_terms:
            synonyms = [s for s in synonyms if s not in avoid_terms]
        if not synonyms:
            return text

        replacement = random.choice(synonyms)
        pattern = re.compile(re.escape(term), re.IGNORECASE)

        def replace_match(match):
            original = match.group(0)
            if original.isupper():
                return replacement.upper()
            elif original[0].isupper():
                return replacement.capitalize()
            return replacement

        return pattern.sub(replace_match, text, count=1)


class MeSHSynonymAugmentor:
    """Replace medical terms with MeSH synonyms."""

    def __init__(self, mesh_extractor: MeSHSynonymExtractor):
        self.mesh = mesh_extractor

    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Candidate terms: capitalised words / phrases plus a few common lowercase
        medical words, filtered to those that have MeSH synonyms.
        """
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        words.extend(re.findall(
            r'\b(?:hypertension|diabetes|pneumonia|infection|disease|syndrome|disorder)\b',
            text, re.IGNORECASE))
        medical_terms = [w for w in words if self.mesh.get_synonyms(w)]
        return list(set(medical_terms))

    def augment(self, retrieved_sources: List[Dict], max_replacements: int = 3,
                text_field: str = 'text') -> List[Dict]:
        """Synonym-replaced copies of the sources (sources without replacements are dropped)."""
        augmented = []
        for source in retrieved_sources:
            if text_field not in source:
                continue
            text = source[text_field]
            medical_terms = self.extract_medical_terms(text)
            if not medical_terms:
                continue

            augmented_text = text
            replaced_terms = []
            for term in medical_terms[:max_replacements]:
                new_text = self.mesh.replace_with_synonym(
                    augmented_text, term, avoid_terms=set(replaced_terms))
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

    def augment_single(self, source: Dict, max_replacements: int = 3,
                       text_field: str = 'text') -> Dict:
        results = self.augment([source], max_replacements, text_field)
        return results[0] if results else source


# =====================================================================
# Term lexicons in JSONL / EuroVoc xlsx (EurLex, arXiv-CS GPT lexicon)
# =====================================================================

def _normalize_term(s: str) -> str:
    """Key used for matching: lower-case, unify dashes, drop spaces and hyphens."""
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


class LexiconSynonymExtractor:
    """
    Term lexicon whose replace_with_synonym() rewrites any variant (term or synonym)
    into the canonical term of its concept (synonym -> term normalisation).

    mode="gpt_jsonl"   : one JSON per line {"term": "...", "synonyms": ["...", ...]}
                         (built by lexicon/build-*-lexicon.py); canonical = "term".
    mode="eurovoc_xlsx": EuroVoc Excel export (ID, TERMS (PT-NPT), PT columns).

    fast_match=False reproduces the paper runs: every lexicon form is checked as a
    normalised substring of the text. fast_match=True uses one word-boundary regex
    instead; it is much faster but matches a slightly different set of terms, so
    the augmented texts (and results) differ from the paper.
    """

    def __init__(self, path: str, mode: str = "gpt_jsonl", sheet_name=0, fast_match: bool = False):
        self.path = Path(path)
        self.mode = mode
        self.sheet_name = sheet_name
        self.fast_match = fast_match

        # concept id -> all surface forms
        self.concept_to_terms: Dict[str, Set[str]] = {}
        # concept id -> canonical term
        self.concept_to_canonical: Dict[str, str] = {}
        # normalised form -> concept ids
        self.termkey_to_concepts: Dict[str, Set[str]] = {}
        # raw strings searched in the text
        self.all_terms_for_match: Set[str] = set()

        if mode == "gpt_jsonl":
            self._load_from_jsonl()
        elif mode == "eurovoc_xlsx":
            self._load_from_xlsx()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # avoid matching very short tokens
        self.all_terms_for_match = {t for t in self.all_terms_for_match if len(t) >= 2}
        self._match_keys = {t: _normalize_term(t) for t in self.all_terms_for_match}
        self._pattern = self._build_pattern() if fast_match else None

        print("=" * 60)
        print(f"Loaded lexicon ({mode}) from: {self.path}")
        print(f"✓ Concepts loaded: {len(self.concept_to_terms)}")
        print(f"✓ Forms indexed: {len(self.all_terms_for_match)}  (fast_match={fast_match})")
        print("=" * 60)

    # ---------------- loading ----------------

    def _add_concept(self, cid: str, terms: Set[str], canonical: str):
        terms = {t for t in terms if t}
        if not terms:
            return
        self.concept_to_terms[cid] = terms
        self.concept_to_canonical[cid] = canonical if canonical is not None else next(iter(terms))
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
                if not term:
                    continue
                cid_counter += 1

                terms_here: Set[str] = set(_variants_for(term))
                for s in obj.get("synonyms") or []:
                    s = (s or "").strip()
                    if s:
                        terms_here.update(_variants_for(s))
                self._add_concept(f"gpt_{cid_counter}", terms_here, canonical=term)

    def _load_from_xlsx(self):
        import pandas as pd

        df = pd.read_excel(self.path, sheet_name=self.sheet_name, dtype=str).fillna("")
        col_id = [c for c in df.columns if str(c).strip().lower() == "id"][0]
        col_terms = [c for c in df.columns if "term" in str(c).lower()][0]  # TERMS (PT-NPT)
        col_pt = [c for c in df.columns if str(c).strip().lower() == "pt"]
        col_pt = col_pt[0] if col_pt else None

        by_id: Dict[str, Set[str]] = {}
        for _, row in df.iterrows():
            cid = row[col_id].strip()
            if not cid:
                continue
            by_id.setdefault(cid, set())
            term = row[col_terms].strip()
            pt = row[col_pt].strip() if col_pt else ""
            if term:
                by_id[cid].update(_variants_for(term))
            if pt:
                by_id[cid].update(_variants_for(pt))

        for cid, terms in by_id.items():
            # canonical: an arbitrary form of the concept
            self._add_concept(cid, terms, canonical=None)

    def _build_pattern(self):
        escaped = sorted((re.escape(t) for t in self.all_terms_for_match), key=len, reverse=True)
        if not escaped:
            return re.compile(r"(?!x)x")
        return re.compile("|".join(rf"\b{e}\b" for e in escaped), flags=re.IGNORECASE)

    # ---------------- lookup ----------------

    def find_terms_in_text(self, text: str) -> List[str]:
        """Lexicon forms occurring in `text`, longest first."""
        if self.fast_match:
            hits = {}
            for m in self._pattern.finditer(text):
                span = m.group(0)
                hits.setdefault(span.lower(), span)
            return sorted(hits.values(), key=lambda s: (-len(s), s.lower()))

        text_key = _normalize_term(text.lower())
        hits: Set[str] = set()
        for t, key in self._match_keys.items():
            if key and key in text_key:
                if re.search(re.escape(t), text, flags=re.IGNORECASE):
                    hits.add(t)
        return sorted(hits, key=lambda s: (-len(s), s.lower()))

    def get_synonyms(self, term: str) -> List[str]:
        """Other forms of the concept(s) of `term`."""
        key = _normalize_term(term)
        if not key or key not in self.termkey_to_concepts:
            return []
        syns: Set[str] = set()
        for cid in self.termkey_to_concepts[key]:
            for t in self.concept_to_terms.get(cid, []):
                if _normalize_term(t) != key:
                    syns.add(t)
        return sorted(syns, key=lambda s: (-len(s), s.lower()))

    def get_canonical(self, term: str) -> Optional[str]:
        """Canonical term of the concept that `term` belongs to (first one if several)."""
        key = _normalize_term(term)
        if not key or key not in self.termkey_to_concepts:
            return None
        cid = next(iter(self.termkey_to_concepts[key]))
        return self.concept_to_canonical.get(cid)

    def replace_with_synonym(self, text: str, term: str, avoid_terms: Set[str] = None) -> str:
        """Rewrite the first occurrence of `term` into its canonical term (case-preserving)."""
        avoid_terms = avoid_terms or set()
        canonical = self.get_canonical(term)
        if not canonical or canonical in avoid_terms:
            return text

        pat = re.compile(re.escape(term), flags=re.IGNORECASE)

        def _repl(m):
            orig = m.group(0)
            if orig.isupper():
                return canonical.upper()
            if orig and orig[0].isupper():
                return canonical[:1].upper() + canonical[1:]
            return canonical

        return pat.sub(_repl, text, count=1)


class LexiconSynonymAugmentor:
    def __init__(self, extractor: LexiconSynonymExtractor):
        self.lex = extractor

    def augment(self, retrieved_sources: List[Dict], max_replacements: int = 3,
                text_field: str = 'text') -> List[Dict]:
        augmented = []
        for src in retrieved_sources:
            if text_field not in src:
                continue
            text = src[text_field]
            terms = self.lex.find_terms_in_text(text)
            if not terms:
                continue
            aug_text = text
            replaced = []
            for term in terms[:max_replacements]:
                new_text = self.lex.replace_with_synonym(aug_text, term, avoid_terms=set(replaced))
                if new_text != aug_text:
                    replaced.append(term)
                    aug_text = new_text
            if replaced:
                item = src.copy()
                item[text_field] = aug_text
                item['augmentation_type'] = 'synonym_to_term_canonicalization'
                item['replaced_terms'] = replaced
                augmented.append(item)
        return augmented

    def augment_single(self, source: Dict, max_replacements: int = 3,
                       text_field: str = 'text') -> Dict:
        results = self.augment([source], max_replacements, text_field)
        return results[0] if results else source


# =====================================================================
# CSO lexicon (arXiv-CS, paper results)
# =====================================================================

class CSOWikidataSynonymAugmentor:
    """
    Synonym replacement with the CSO lexicon (cs_wikidata_lexicon_from_cso.json):

    {
      "0-1 knapsack problem": {
        "wikidata_id": "Q105415774",
        "forms": ["0-1 knapsack problem", "binary knapsack problem"],
        ...
      },
      ...
    }

    Also accepts the cso_synonyms_clean.json format
    {"nonrigid registration": {"preferred": "...", "synonyms": [...]}, ...}.

    Terms are matched with a single word-boundary regex and replaced by a random
    other form of the same concept.
    """

    def __init__(self, lexicon_path: str):
        self.lexicon_path = Path(lexicon_path)
        # concept id -> all forms (lower-cased)
        self.concept_to_synonyms: Dict[str, List[str]] = {}
        # form (lower) -> concept id
        self.term_to_concept: Dict[str, str] = {}

        print("=" * 60)
        print(f"Loading CSO lexicon from: {self.lexicon_path}")
        print("=" * 60)
        self._load_lexicon()
        self.term_pattern = self._build_pattern()
        print(f"✓ Total concepts loaded: {len(self.concept_to_synonyms)}")
        print(f"✓ Total terms indexed: {len(self.term_to_concept)}")
        print("=" * 60)

    def _load_lexicon(self):
        with self.lexicon_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for outer_key, entry in data.items():
            preferred = entry.get("preferred") or outer_key
            if entry.get("forms"):
                forms = list(entry["forms"])
            else:
                forms = [preferred] + list(entry.get("synonyms") or [])

            forms_lower = sorted({s.strip().lower() for s in forms if isinstance(s, str) and s.strip()})
            if not forms_lower:
                continue
            self.concept_to_synonyms[preferred] = forms_lower
            for s in forms_lower:
                # a form keeps pointing to the first concept it was seen in
                self.term_to_concept.setdefault(s, preferred)

    def _build_pattern(self):
        # longest first, so that short terms do not cut long ones
        escaped = sorted((re.escape(t) for t in self.term_to_concept), key=len, reverse=True)
        if not escaped:
            return re.compile(r"(?!x)x")
        return re.compile("|".join(rf"\b{e}\b" for e in escaped), flags=re.IGNORECASE)

    def get_synonyms(self, term: str) -> List[str]:
        key = term.lower()
        concept = self.term_to_concept.get(key)
        if not concept:
            return []
        return [s for s in self.concept_to_synonyms[concept] if s != key]

    def replace_with_synonym(self, text: str, term: str, avoid_terms: Set[str] = None) -> str:
        """Replace the first occurrence of `term` with a random synonym (case-preserving)."""
        synonyms = self.get_synonyms(term)
        if avoid_terms:
            avoid_lower = {t.lower() for t in avoid_terms}
            synonyms = [s for s in synonyms if s not in avoid_lower]
        if not synonyms:
            return text

        replacement = random.choice(synonyms)
        pattern = re.compile(re.escape(term), re.IGNORECASE)

        def replace_match(match):
            original = match.group(0)
            if original.isupper():
                return replacement.upper()
            elif original[0].isupper():
                return replacement.capitalize()
            return replacement

        return pattern.sub(replace_match, text, count=1)

    def find_terms_in_text(self, text: str) -> List[str]:
        """Replaceable terms in `text` (as written in the text), de-duplicated."""
        terms: List[str] = []
        seen: Set[str] = set()
        for m in self.term_pattern.finditer(text):
            span = m.group(0)
            key = span.lower()
            if key in seen:
                continue
            if self.get_synonyms(span):
                seen.add(key)
                terms.append(span)
        return terms

    def augment(self, retrieved_sources: List[Dict[str, Any]], max_replacements: int = 3,
                text_field: str = "text") -> List[Dict[str, Any]]:
        augmented: List[Dict[str, Any]] = []
        for source in retrieved_sources:
            text = source.get(text_field)
            if not text:
                continue
            terms = self.find_terms_in_text(text)
            if not terms:
                continue

            augmented_text = text
            replaced_terms: List[str] = []
            replaced_terms_lc: Set[str] = set()
            for term in terms[:max_replacements]:
                new_text = self.replace_with_synonym(augmented_text, term, avoid_terms=replaced_terms_lc)
                if new_text != augmented_text:
                    augmented_text = new_text
                    replaced_terms.append(term)
                    replaced_terms_lc.add(term.lower())

            if replaced_terms:
                aug_sample = dict(source)
                aug_sample[text_field] = augmented_text
                aug_sample["augmentation_type"] = "cso_synonym_replacement"
                aug_sample["replaced_terms"] = replaced_terms
                augmented.append(aug_sample)
        return augmented

    def augment_single(self, source: Dict[str, Any], max_replacements: int = 3,
                       text_field: str = "text") -> Dict[str, Any]:
        augmented = self.augment([source], max_replacements, text_field)
        return augmented[0] if augmented else dict(source)
