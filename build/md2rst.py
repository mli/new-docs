import sys
import os
import time
import re
# import notedown
import nbformat
import pypandoc

def parse_header(inputs):
    """return the body of the markdown contents, and header key-value pairs
    """
    is_header_mark = lambda x: x.strip() == '---'
    lines = inputs.split('\n')
    num_marks = 0
    for l in lines:
        if is_header_mark(l):
            num_marks += 1
    if num_marks == 0 or not is_header_mark(lines[0]):
        return inputs, {}
    assert num_marks >= 2, 'the header section is not closed'
    i, kv, kv_re = 1, {}, re.compile('^([\w]+)[ ]*\:[ ]*([\w]+)')
    while not is_header_mark(lines[i]):
        m = kv_re.match(lines[i])
        if m is not None:
            kv[m.groups()[0]] = m.groups()[1]
        i += 1
    return '\n'.join(lines[i+1:]), kv

def md2rst():
    assert len(sys.argv) == 3, 'usage: input.md output.rst'
    (src_fn, input_fn, output_fn) = sys.argv

    with open(input_fn, 'r') as f:
        inputs = f.read()

    inputs, kv  = parse_header(inputs)
    if not 'run' in kv or kv['run'].lower() != 'true':
        print('%s: Convert %s to %s directly'  % (src_fn, input_fn, output_fn))
        outputs = pypandoc.convert_text(inputs, 'rst', format='md')
        with open(output_fn, 'w') as f:
            f.write(outputs)
        return

    # timeout for each notebook, in min
    timeout = 20 # in default
    if 'timeout' in kv:
        timeout = int(kv['timeout'])
    print('%s: Start to run %s with timeout %d min' % (src_fn, input_fn, timeout))

    print('%s: Done in %.1f min, write results into %s' % (src_fn, 1, output_fn))

if __name__ == '__main__':
    md2rst()

# # the files will be ingored for execution
# ignore_execution = []


# reader = notedown.MarkdownReader(match='strict')

# do_eval = int(os.environ.get('EVAL', True))

# # read
# with open(input_fn, 'r') as f:
#     notebook = reader.read(f)

# for c in notebook.cells:
#     c.source = add_space_between_ascii_and_non_ascii(c.source)

# if do_eval and not any([i in input_fn for i in ignore_execution]):
#     tic = time.time()
#     notedown.run(notebook, timeout)
#     print('=== Finished evaluation in %f sec'%(time.time()-tic))

# # write
# # need to add language info to for syntax highlight
# notebook['metadata'].update({'language_info':{'name':'python'}})

# with open(output_fn, 'w') as f:
#     f.write(nbformat.writes(notebook))
