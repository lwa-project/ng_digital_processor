# Makefile for NDP Documentation
# Usage:
#   make          - Build all documents
#   make clean    - Remove build artifacts
#   make distclean - Remove all generated files including PDFs

LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Documents
NDP_ICD_DIR = NDP_ICD
NDP_ICD_DOC = NDP_ICD

.PHONY: all clean distclean ndp_icd

# Default target
all: ndp_icd

# NDP ICD document
ndp_icd: $(NDP_ICD_DOC).pdf

$(NDP_ICD_DOC).pdf: $(NDP_ICD_DIR)/lwa-icd.cls \
                    $(NDP_ICD_DIR)/$(NDP_ICD_DOC).tex \
                    $(wildcard $(NDP_ICD_DIR)/config/*.tex) \
                    $(wildcard $(NDP_ICD_DIR)/frontmatter/*.tex) \
                    $(wildcard $(NDP_ICD_DIR)/sections/*.tex) \
                    $(wildcard $(NDP_ICD_DIR)/appendices/*.tex)
	cd $(NDP_ICD_DIR) && $(LATEX) $(LATEX_FLAGS) $(NDP_ICD_DOC).tex
	cd $(NDP_ICD_DIR) && $(LATEX) $(LATEX_FLAGS) $(NDP_ICD_DOC).tex
	mv $(NDP_ICD_DIR)/$(NDP_ICD_DOC).pdf .

# Clean build artifacts
clean:
	cd $(NDP_ICD_DIR) && rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	cd $(NDP_ICD_DIR) && rm -f sections/*.aux appendices/*.aux frontmatter/*.aux config/*.aux

# Remove all generated files including PDFs
distclean: clean
	rm -f $(NDP_ICD_DOC).pdf

# Help
help:
	@echo "NDP Documentation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all documents (default)"
	@echo "  ndp_icd    - Build NDP ICD document"
	@echo "  clean      - Remove build artifacts"
	@echo "  distclean  - Remove all generated files including PDFs"
	@echo "  help       - Show this help message"
