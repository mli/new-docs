all: html

# markdown files that will not be evaluated, simply copy to build/
PURE_MARKDOWN = ./README.md
# markdown files that will be evaluated and then saved as ipynb files
IPYNB_MARKDOWN = $(shell find . -not -path "./build/*" -name "*.md")
# RST files will be simply coped to build/
RST = $(shell find . -not -path "./build/*" -name "*.rst")

OBJ = $(patsubst %.rst, build/%.rst, $(RST)) \
	$(patsubst %.md, build/%.md, $(PURE_MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, \
		$(filter-out $(PURE_MARKDOWN), $(IPYNB_MARKDOWN)))

build/%.ipynb: %.md
	@mkdir -p $(@D)
	python build/md2ipynb.py $< $@


build/%.rst: %.rst
	@mkdir -p $(@D)
	python build/process_rst.py $< $@

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(OBJ)
	sphinx-autogen build/api/*.rst build/api/*/*.rst   -t build/_templates/
	# make -C build linkcheck doctest html
	make -C build html


clean:
	rm -rf build/_build $(OBJ) build/api build/develop api/_autogen api/*/_autogen
