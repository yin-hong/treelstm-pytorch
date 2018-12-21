import os


def make_dir(dir):
    """
    make dir if dir is not found
    :param dir: dir name
    :return:
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def build(dirpath, cp=''):
    """
    build the java file
    :param dirpath: dir name of java file
    :param cp: jar path
    :return:
    """
    print('\n Build class')
    cmd = (' javac -cp $' + cp + ' ' +dirpath + '/*.java')
    os.system(cmd)


def dependency_parse(filepath, cp='', tokenize=True):
    """
    dependency parse
    :param filepath: one sentence per line in this file
    :param cp: jar path
    :param tokenize: keep the sentences after parsing
    :return:
    """
    dirname = os.path.dirname(filepath)   # get the dir name
    basename = os.path.basename(filepath)
    base = os.path.splitext(basename)[0]
    parentpath = os.path.join(dirname, base + '.parents')
    relpath = os.path.join(dirname, base + '.rels')
    tokpath = os.path.join(dirname, base + '.toks')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
             % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)


def constituency_parse(filepath, cp='', tokenize=True):
    """
    constituency parse
    :param filepath: one sentence per line in this file
    :param cp: jar path
    :param tokenize: keep the sentences after parsing
    :return:
    """
    dirname = os.path.dirname(filepath)   # get the dir name
    basename = os.path.basename(filepath)
    base = os.path.splitext(basename)[0]
    parentpath = os.path.join(dirname, base + '.cparents')
    tokpath = os.path.join(dirname, base + '.toks')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
             % (cp, tokpath, parentpath,  tokenize_flag, filepath))
    os.system(cmd)


def split(filepath, dst_dir):
    """
    split the original SICK dataset file into id, sent_a, sent_b, sim
    :param filepath: SICK dataset file
    :param dst_dir: destination dir
    :return:
    """
    file_id = os.path.join(dst_dir, 'id.txt')
    file_sent_a = os.path.join(dst_dir, 'a.txt')
    file_sent_b = os.path.join(dst_dir, 'b.txt')
    file_sim = os.path.join(dst_dir, 'sim.txt')
    with open(filepath, 'r', encoding='utf8', errors='ignore') as f, \
           open(file_id, 'w') as f_id, \
           open(file_sent_a, 'w') as f_a, \
           open(file_sent_b, 'w') as f_b, \
           open(file_sim, 'w') as f_s:
        line = f.readline()  # skip the first line
        for line in f:
            contents = line.strip().split('\t')
            f_id.write(contents[0] + '\n')
            f_a.write(contents[1] + '\n')
            f_b.write(contents[2] + '\n')
            f_s.write(contents[3] + '\n')


def parse(dirpath, cp='', tokenize=True):
    """
    parse sentences, including dependency parsing, constituency parse
    :param dirpath:contain sentence_a file, sentenc_b file.
    :param cp:jar path
    :param tokenize:keep the sentences after parsing
    :return:
    """
    dependency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=tokenize)
    dependency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=tokenize)
    constituency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=tokenize)
    constituency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=tokenize)


if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing SICK Dataset')
    dir_name = '../data/sick/'
    train_dir = os.path.join(dir_name, 'train/')
    make_dir(train_dir)
    dev_dir = os.path.join(dir_name, 'dev/')
    make_dir(dev_dir)
    test_dir = os.path.join(dir_name, 'test/')
    make_dir(test_dir)
    split(os.path.join(dir_name, 'SICK_train.txt'), train_dir)
    split(os.path.join(dir_name, 'SICK_trial.txt'), dev_dir)
    split(os.path.join(dir_name, 'SICK_test_annotated.txt'), test_dir)
    lib_dir = '../lib/'
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    build(lib_dir, cp=classpath)
    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)
    parse(test_dir, cp=classpath)



