import os, argparse
parser = argparse.ArgumentParser()

def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Program Parameters")
    parser.add_argument(
            '--dname',
            type=str,
            default='unknown'
            help='the name of the dataset: stg, pzv, wsht. This will overwrite ' + \
                 'the arg of n_removed_first_chars in the case the dname is in the,' + \
                 'known datasets, as it depends on the dataset...'
    )
    parser.add_argument(
            '--folder_path',
            type=str,
            default=".",
            help='the folder there is the file to process (containing the whole ' + \
                 'transcriptions of the data (e.g., transcriptions.txt)'
    )
    parser.add_argument(
            '-i',
            '--in_file_name',
            type=str,
            help='the name of the file to process (where there are the raw ' + \
                 'transcriptions)'
    )
    parser.add_argument(
            '-o',
            '--out_file_name',
            type=str,
            default='special_character_codes.txt',
            help='the name of the output file'
    )
    parser.add_argument(
            '--char_list',
            type=str,
            default=['\n','|','-'],
            nargs="+",
            help='the first three characters are taken as delimiters of the characters'
    )
    parser.add_argument(
            '--n_removed_first_chars',
            type=int,
            default=0, #14 for stg, 11 for pzv, 7 for wsht
            help='n. of first characters to be removed'
    )

    return parser


parser = parse_input_arguments()
args = parser.parse_args()


with open(os.path.join(args.folder_path, args.in_file_name), 'r') as f_in:
    with open(os.path.join(args.folder_path, args.out_file_name), 'w+') as f_out:
        codices_set = set()
        
        for line in f_in:
            # we remove the first characters that contain the id
            if args.dname == 'stg':
                args.n_removed_first_chars = 14
                # the saint gall dataset has both the transcription at character level
                # and at word level in the same file
                line = line.split(' ')[-2]
            elif args.dname == 'pzv':
                args.n_removed_first_chars = 11
            elif args.dname == 'wsht':
                args.n_removed_first_chars = 7
            else:
                print('The dataset name used is not managed by this script!')
                print('The number of characters removed from the start of the lines' + \
                      ' is the one passed or the default one (0)...')
            
            line = line[args.n_removed_first_chars:]
           
            if args.dname == 'stg':
                line = line.split(' ')[0]
            pos_char0 = [pos for pos,char in enumerate(line) if char==args.char_list[0]]
            pos_char1 = [pos for pos,char in enumerate(line) if char==args.char_list[1]]
            pos_char2 = [pos for pos,char in enumerate(line) if char==args.char_list[2]]
            # we collect all the positions of the delimiters of characters 
            # and words in a line
            all_pos = pos_char0 + pos_char1 + pos_char2
            all_pos = sorted(all_pos)
            for i in range(1, len(all_pos)):
                # 0 (delimiter) 1 (char) 2 (delimiter)-> 2-0=2 and it is the case
                # of having a common character
                if all_pos[i]-all_pos[i-1]>2:
                    # we must esclude the positions of the delimiters themselves
                    codices_set |= set([line[all_pos[i-1]+1:all_pos[i]]])
        output = sorted([x for x in codices_set])
        f_out.write('\n'.join(output))
        f_out.write('\n')