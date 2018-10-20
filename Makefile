all: html


MARKDOWN = $(shell find . -name "*.md")
RST = $(shell find . -name "*.rst")

OBJ = $(patsubst %.rst, build/%.rst, $(RST)) \
	$(patsubst %.md, build/%.rst, $(MARKDOWN))

build/%.rst: %.md
	@mkdir -p $(@D)
	python build/md2rst.py $< $@


build/%.rst: %.rst
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(OBJ)
	echo $(OBJ)
	# make -C build html

clean:
	rm -rf build/chapter* build/_build $(DEPS)
