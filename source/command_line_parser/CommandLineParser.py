import argparse


class CommandLineParser(object):

    def parse(self):

        cli_parser = argparse.ArgumentParser()
        cli_parser.add_argument("-cp", "--contentpath", help="Content Image Path.")
        cli_parser.add_argument("-sp", "--stylepath", help="Style Image Path.")
        cli_parser.add_argument("-i", "--iterations", help="Number of iterations.")
        cli_parser.add_argument("-cw", "--contentweight", help="Content Weight.")
        cli_parser.add_argument("-sw", "--styleweight", help="Style Weight.")
        cli_parser.add_argument("-w", "--imagewidth", help="Max image width.")

        args = cli_parser.parse_args()
        target_args = {}

        print(
            "content image path: {}, "
            "style image path: {}, "
            "number of iterations: {}, "
            "content weight: {}, "
            "style weight: {}, "
            "target image width: {}"
                .format(
                args.contentpath,
                args.stylepath,
                args.iterations,
                args.contentweight,
                args.styleweight,
                args.imagewidth
                )
        )

        if args.contentpath is None:
            raise TypeError
        else:
            target_args["content_path"] = args.contentpath

        if args.stylepath is None:
            raise TypeError
        else:
            target_args["style_path"] = args.stylepath

        if args.iterations is None:
            target_args["iterations"] = 1000
        else:
            target_args["iterations"] = int(args.iterations)

        if args.contentweight is None:
            target_args["content_weight"] = 1e3
        else:
            target_args["content_weight"] = float(args.contentweight)

        if args.styleweight is None:
            target_args["style_weight"] = 1e-2
        else:
            target_args["style_weight"] = float(args.styleweight)

        if args.imagewidth is None:
            target_args["target_image_width"] = 1024
        else:
            target_args["target_image_width"] = int(args.imagewidth)

        return target_args.values()
