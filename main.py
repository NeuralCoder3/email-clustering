import optparse
import os
from clustering import preprocess
from clustering import cluster


def main():
    parser = optparse.OptionParser()
    parser.add_option("-p", "--preprocess", action="store_true", dest="preprocess", default=True,
                      help="Preprocess E-Mails (extract relevant meta data and render HTML to plain text)")
    parser.add_option("-n", "--number", action="store", dest="number", default=6,
                      help="Number of clusters to create")
    parser.add_option("-i", "--input", action="store", dest="input", default="./mails",
                      help="Input folder for Texts/E-Mails")
    parser.add_option("-s", "--stopwords", action="store", dest="stopwords", default="custom_stopwords2.txt",
                      help="Additional stopword list to use")
    parser.add_option("-o", "--output", action="store", dest="output", default="./clusters",
                      help="Output folder for clusters")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                      help="Verbose output")
    parser.add_option("-e", "--estimate", action="store_true", dest="estimate", default=False,
                      help="Estimate number of clusters")

    (options, args) = parser.parse_args()

    if options.preprocess:
        tmp_folder = options.input.strip("/") + "_preprocessed"
        if not os.path.exists(tmp_folder):
            preprocess.preprocess(options.input, tmp_folder)
        options.input = tmp_folder

    cluster.cluster(options.number, options.input, options.output,
                    [options.stopwords], verbose=options.verbose, compute_variance=options.estimate)


if __name__ == "__main__":
    main()
