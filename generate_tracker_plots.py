import matplotlib

matplotlib.use('Agg')

from dst_tools import get_consumer_trust
import plotting
import utils


def shout(s):
    width = 42
    c = "*"
    s = "  " + s + "  "
    while len(s) + 2 < width:
        s = c+s+c
    while len(s) < 42:
        s += c
    sep = width*c
    lines = ["", sep, s, sep, ""]
    msg = "\n".join(lines)
    print(msg)


def main():
    shout("Updating Twitter data")
    utils.update_all()
    df_sentiment = utils.read_twitter_sentiment()
    df_cti = get_consumer_trust()

    shout("Generating plot")
    fig, _ = plotting.plot_covid_impact(df_sentiment, df_cti)
    plotting.save(fig, "eco_sentiment.png")


if __name__ == '__main__':
    main()
