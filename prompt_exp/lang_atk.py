from nltk.corpus import wordnet
from transformers import AutoTokenizer, CLIPTextModel
import torch
import numpy as np
import itertools
from utils import encode_text, test_rewrite, EVAL_COLORS

lang_template = "put the {pick} blocks in a {place} bowl"
original_sentence = lang_template.format(pick="green", place="blue")
target_sentence = lang_template.format(pick="blue", place="green")
all_tasks = [lang_template.format(pick=c1, place=c2) \
                        for c1, c2 in itertools.product(EVAL_COLORS, EVAL_COLORS)]
target_idx = all_tasks.index(target_sentence)

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

atk_sentences = [
    "place the green cubes into a cerulean container",
    "insert the verdant blocks into a sapphire bowl",
    "toss the lime blocks into a navy basin",
    "arrange the jade squares in a teal receptacle",
    "deposit the emerald blocks within an azure dish",
    "drop the forest green bricks into a cobalt vessel",
    "shift the mint blocks into a lapis bowl",
    "slide the sea green blocks into a marine blue bowl",
    "set the pine green tiles in a sky blue container",
    "load the grass green pieces into an indigo bowl",
    "transfer the olive blocks to a turquoise vessel",
    "assemble the shamrock green cubes in a blueberry-colored bowl",
    "stack the kelly green blocks in a peacock blue dish",
    "place the leaf green bricks into a steel blue pot",
    "situate the mossy green cubes within a robin's egg blue basin",
    "organize the pastel green blocks in a royal blue receptacle",
    "gather the neon green squares into a midnight blue vessel",
    "stow the bright green tiles in a powder blue container",
    "put the pale green bricks into a deep blue bowl",
    "position the chartreuse blocks in a light blue receptacle",
    "nestle the fern green pieces into a dark blue vessel",
    "insert the hunter green blocks into an ice blue dish",
    "tuck the pear green cubes into a denim blue bowl",
    "fit the apple green bricks into a navy blue basin",
    "align the wasabi blocks in a prussian blue receptacle",
    "convey the spruce green tiles into a Persian blue container",
    "introduce the malachite green pieces to a cornflower blue dish",
    "locate the viridian blocks inside a sapphire blue bowl",
    "escort the sage green cubes to a Capri blue vessel",
    "guide the algae green bricks to an ultramarine basin",
    "maneuver the pistachio green blocks into an electric blue receptacle",
    "usher the crocodile green squares into an iris blue bowl",
    "shuffle the willow green pieces into a baby blue dish",
    "relocate the avocado green tiles to a steel blue container",
    "direct the seaweed green blocks into an Oxford blue bowl",
    "compile the olive drab bricks inside a Mediterranean blue receptacle",
    "transport the spinach green cubes to a blue ice vessel",
    "corral the pickle green blocks into a tealish bowl",
    "channel the forest shade bricks into a cobaltish dish",
    "sweep the parakeet green squares into a bluish basin",
    "herd the beryl green tiles into a ceruleanish receptacle",
    "funnel the harlequin green pieces into a blue-toned vessel",
    "round up the pea green blocks into a pale blue bowl",
    "accumulate the celadon tiles in a deep bluish dish",
    "nudge the spearmint green blocks into a darkish blue container",
    "coax the camouflage green bricks to a lightish blue vessel",
    "migrate the myrtle green cubes into a sapphire-ish bowl",
    "ease the thyme green blocks into a navyish basin",
    "wheedle the broccoli green pieces into a royalish blue dish",
    "entice the alligator green squares into a skyish blue receptacle",
    "marshal the juniper green tiles to an indigoish container",
    "congregate the zucchini green blocks inside a teal-blue bowl",
    "amass the minty bricks into a cobalt-blue dish",
    "consolidate the turtle green cubes into a marine-blue basin",
    "ferry the clover green blocks to an azure-blue vessel",
    "herald the asparagus green pieces into a peacock-blue container",
    "bid the kiwi green squares into a denim-blue bowl",
    "summon the eucalyptus green tiles into an iris-blue dish",
    "beckon the basil green bricks to an Oxford-blue basin",
    "invite the fernlike blocks into a Mediterranean-blue receptacle",
    "lure the lichen green cubes to a Persian-blue bowl",
    "motion the bay green bricks into a Capri-blue vessel",
    "draw the holly green blocks into a cornflower-blue dish",
    "enlist the pine needle green pieces in a powder-blue container",
    "woo the artichoke green squares into a baby-blue bowl",
    "steer the swamp green tiles to a deep-blue receptacle",
    "shepherd the army green bricks into a dark-blue vessel",
    "guide the meadow green cubes into an electric-blue bowl",
    "prompt the forest moss blocks into a royal-blue dish",
    "urge the sagebrush green tiles into a light-blue basin",
    "incline the kelp green bricks into a steel-blue vessel",
    "channel the fern frond blocks into an ultramarine bowl",
    "propel the seafoam green cubes into a tealish-blue receptacle",
    "leverage the pickle rind green pieces to a bluish dish",
    "advance the tarragon green blocks into a blue-toned bowl",
    "route the cactus green tiles to a deep bluish vessel",
    "deploy the moss bank bricks into a darkish blue container",
    "distribute the algae pond blocks into a lightish blue receptacle",
    "relay the kiwi skin cubes to a sapphire-ish dish",
    "dispatch the garden green bricks into a navyish bowl",
    "project the cricket green blocks into a royalish blue basin",
    "scoot the bamboo green tiles into a skyish blue dish"
]


for atk_sentence in atk_sentences:
    test_rewrite(model, tokenizer, atk_sentence, all_tasks, target_idx)