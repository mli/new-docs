all: html

# markdown files that will not be evaluated, simply copy to build/
PURE_MARKDOWN =  ./README.md \
                 $(wildcard ./blog/*.md) \
                 $(wildcard ./install/*.md)
# markdown files that will be evaluated and then saved as ipynb files
IPYNB_MARKDOWN = $(shell find . -not -path "./build/*" -name "*.md" | sort -h)
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
	# python build/process_rst.py $< $@
	cp $< $@

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

#debug:
# 	@echo $(PURE_MARKDOWN)
#	@echo $(IPYNB_MARKDOWN)
# 	@echo $(filter-out $(PURE_MARKDOWN), $(IPYNB_MARKDOWN))

html: $(OBJ)
	sphinx-autogen build/api/*.rst build/api/*/*.rst   -t build/_templates/
	# make -C build linkcheck doctest html
	make -C build html
	sed -i.bak 's/33\,150\,243/23\,141\,201/g'  build/_build/html/_static/material-design-lite-1.3.0/material.blue-deep_orange.min.css


clean:
	rm -rf build/_build $(OBJ) build/api build/develop api/_autogen api/*/_autogen
