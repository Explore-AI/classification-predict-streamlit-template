from wordcloud import WordCloud, STOPWORDS
import codecs
import imageio
import resources.base_string as img
# visualisation
from matplotlib import pyplot as plt

def gen_wordcloud(phrases, title, savefig=False):
    freqs = [(phrase, freq) for phrase, freq in phrases.items()]
    img_str = img.base_img()
    tw = open("tweet_bird.jpg", "wb")
    tw.write(codecs.decode(img_str,'base64'))
    tw.close()
    tw_mask = imageio.imread('tweet_bird.jpg')
    
    plt.figure(figsize=(16, 12))
    wc = WordCloud(colormap='magma', mask=tw_mask, background_color=None, max_words=10_000, mode="RGBA")
    wc.generate_from_frequencies(phrases)
    plt.title(title, fontsize=24)
    plt.axis("off")
    plt.margins(tight=True)
    plt.imshow(wc, interpolation="bilinear")
    if savefig:
        plt.savefig(f'{title}.png', transparent=True, bbox_inches='tight', dpi=200)
    plt.show()
    