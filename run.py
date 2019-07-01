import argparse

from Coursera.Exercise1 import ex1, ex1_multi
from Coursera.Exercise2 import ex2, ex2_reg
from Coursera.Exercise3 import ex3, ex3_nn
from Coursera.Exercise4 import ex4
from Coursera.Exercise5 import ex5
from Coursera.Exercise6 import ex6, ex6_spam
from Coursera.Exercise7 import ex7, ex7_pca
from Coursera.Exercise8 import ex8, ex8_movies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('script_type', help="server | coverage")
    parser.add_argument('--out', help="output file path")
    args = parser.parse_args()
    script_type = args.script_type

    if script_type == 'ex1':
        ex1.run()

    elif script_type == 'ex1_multi':
        ex1_multi.run()

    elif script_type == 'ex2':
        ex2.run()

    elif script_type == 'ex2_reg':
        ex2_reg.run()

    elif script_type == 'ex3':
        ex3.run()

    elif script_type == 'ex3_nn':
        ex3_nn.run()

    elif script_type == 'ex4':
        ex4.run()

    elif script_type == 'ex5':
        ex5.run()

    elif script_type == 'ex6':
        ex6.run()

    elif script_type == 'ex6_spam':
        ex6_spam.run()

    elif script_type == 'ex7':
        ex7.run()

    elif script_type == 'ex7_pca':
        ex7_pca.run()

    elif script_type == 'ex8':
        ex8.run()

    elif script_type == 'ex8_movies':
        ex8_movies.run()

    else:
        print("invalid script name - ex1 | ex1_multi | ex2 | ex2_reg | ex3 | ex3_nn | ex4 |ex5 | "
              "ex6 | ex6_spam | ex7 | ex7_pca | ex8 | ex8_movies")
