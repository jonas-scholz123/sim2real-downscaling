# %%
from typing import Tuple
import urllib.request
import os
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile
import geopandas as gpd
import numpy as np

from sim2real.config import data, paths, names
from sim2real.plots import init_fig
from sim2real.utils import ensure_dir_exists
from sim2real.datasets import DWDStationData, load_station_splits

VALUE_STATIONS_FPATH = f"{paths.root}/helper_files/value_stations.txt"


fnames = [
    "TU_Stundenwerte_Beschreibung_Stationen.txt",
    "stundenwerte_TU_00044_akt.zip",
    "stundenwerte_TU_00073_akt.zip",
    "stundenwerte_TU_00078_akt.zip",
    "stundenwerte_TU_00091_akt.zip",
    "stundenwerte_TU_00096_akt.zip",
    "stundenwerte_TU_00102_akt.zip",
    "stundenwerte_TU_00125_akt.zip",
    "stundenwerte_TU_00131_akt.zip",
    "stundenwerte_TU_00142_akt.zip",
    "stundenwerte_TU_00150_akt.zip",
    "stundenwerte_TU_00151_akt.zip",
    "stundenwerte_TU_00154_akt.zip",
    "stundenwerte_TU_00161_akt.zip",
    "stundenwerte_TU_00164_akt.zip",
    "stundenwerte_TU_00167_akt.zip",
    "stundenwerte_TU_00183_akt.zip",
    "stundenwerte_TU_00191_akt.zip",
    "stundenwerte_TU_00198_akt.zip",
    "stundenwerte_TU_00217_akt.zip",
    "stundenwerte_TU_00222_akt.zip",
    "stundenwerte_TU_00232_akt.zip",
    "stundenwerte_TU_00257_akt.zip",
    "stundenwerte_TU_00259_akt.zip",
    "stundenwerte_TU_00282_akt.zip",
    "stundenwerte_TU_00294_akt.zip",
    "stundenwerte_TU_00298_akt.zip",
    "stundenwerte_TU_00303_akt.zip",
    "stundenwerte_TU_00314_akt.zip",
    "stundenwerte_TU_00320_akt.zip",
    "stundenwerte_TU_00330_akt.zip",
    "stundenwerte_TU_00342_akt.zip",
    "stundenwerte_TU_00368_akt.zip",
    "stundenwerte_TU_00377_akt.zip",
    "stundenwerte_TU_00379_akt.zip",
    "stundenwerte_TU_00390_akt.zip",
    "stundenwerte_TU_00400_akt.zip",
    "stundenwerte_TU_00403_akt.zip",
    "stundenwerte_TU_00420_akt.zip",
    "stundenwerte_TU_00427_akt.zip",
    "stundenwerte_TU_00430_akt.zip",
    "stundenwerte_TU_00433_akt.zip",
    "stundenwerte_TU_00445_akt.zip",
    "stundenwerte_TU_00460_akt.zip",
    "stundenwerte_TU_00535_akt.zip",
    "stundenwerte_TU_00555_akt.zip",
    "stundenwerte_TU_00591_akt.zip",
    "stundenwerte_TU_00596_akt.zip",
    "stundenwerte_TU_00603_akt.zip",
    "stundenwerte_TU_00617_akt.zip",
    "stundenwerte_TU_00656_akt.zip",
    "stundenwerte_TU_00662_akt.zip",
    "stundenwerte_TU_00691_akt.zip",
    "stundenwerte_TU_00701_akt.zip",
    "stundenwerte_TU_00704_akt.zip",
    "stundenwerte_TU_00722_akt.zip",
    "stundenwerte_TU_00755_akt.zip",
    "stundenwerte_TU_00757_akt.zip",
    "stundenwerte_TU_00760_akt.zip",
    "stundenwerte_TU_00766_akt.zip",
    "stundenwerte_TU_00769_akt.zip",
    "stundenwerte_TU_00817_akt.zip",
    "stundenwerte_TU_00840_akt.zip",
    "stundenwerte_TU_00850_akt.zip",
    "stundenwerte_TU_00853_akt.zip",
    "stundenwerte_TU_00856_akt.zip",
    "stundenwerte_TU_00860_akt.zip",
    "stundenwerte_TU_00867_akt.zip",
    "stundenwerte_TU_00880_akt.zip",
    "stundenwerte_TU_00891_akt.zip",
    "stundenwerte_TU_00896_akt.zip",
    "stundenwerte_TU_00917_akt.zip",
    "stundenwerte_TU_00953_akt.zip",
    "stundenwerte_TU_00954_akt.zip",
    "stundenwerte_TU_00963_akt.zip",
    "stundenwerte_TU_00979_akt.zip",
    "stundenwerte_TU_00983_akt.zip",
    "stundenwerte_TU_00991_akt.zip",
    "stundenwerte_TU_01001_akt.zip",
    "stundenwerte_TU_01048_akt.zip",
    "stundenwerte_TU_01050_akt.zip",
    "stundenwerte_TU_01051_akt.zip",
    "stundenwerte_TU_01052_akt.zip",
    "stundenwerte_TU_01072_akt.zip",
    "stundenwerte_TU_01078_akt.zip",
    "stundenwerte_TU_01103_akt.zip",
    "stundenwerte_TU_01107_akt.zip",
    "stundenwerte_TU_01161_akt.zip",
    "stundenwerte_TU_01197_akt.zip",
    "stundenwerte_TU_01200_akt.zip",
    "stundenwerte_TU_01207_akt.zip",
    "stundenwerte_TU_01214_akt.zip",
    "stundenwerte_TU_01224_akt.zip",
    "stundenwerte_TU_01228_akt.zip",
    "stundenwerte_TU_01246_akt.zip",
    "stundenwerte_TU_01255_akt.zip",
    "stundenwerte_TU_01262_akt.zip",
    "stundenwerte_TU_01266_akt.zip",
    "stundenwerte_TU_01270_akt.zip",
    "stundenwerte_TU_01279_akt.zip",
    "stundenwerte_TU_01297_akt.zip",
    "stundenwerte_TU_01300_akt.zip",
    "stundenwerte_TU_01303_akt.zip",
    "stundenwerte_TU_01327_akt.zip",
    "stundenwerte_TU_01332_akt.zip",
    "stundenwerte_TU_01339_akt.zip",
    "stundenwerte_TU_01346_akt.zip",
    "stundenwerte_TU_01357_akt.zip",
    "stundenwerte_TU_01358_akt.zip",
    "stundenwerte_TU_01411_akt.zip",
    "stundenwerte_TU_01420_akt.zip",
    "stundenwerte_TU_01424_akt.zip",
    "stundenwerte_TU_01443_akt.zip",
    "stundenwerte_TU_01451_akt.zip",
    "stundenwerte_TU_01468_akt.zip",
    "stundenwerte_TU_01503_akt.zip",
    "stundenwerte_TU_01504_akt.zip",
    "stundenwerte_TU_01526_akt.zip",
    "stundenwerte_TU_01544_akt.zip",
    "stundenwerte_TU_01550_akt.zip",
    "stundenwerte_TU_01572_akt.zip",
    "stundenwerte_TU_01580_akt.zip",
    "stundenwerte_TU_01584_akt.zip",
    "stundenwerte_TU_01587_akt.zip",
    "stundenwerte_TU_01590_akt.zip",
    "stundenwerte_TU_01602_akt.zip",
    "stundenwerte_TU_01605_akt.zip",
    "stundenwerte_TU_01612_akt.zip",
    "stundenwerte_TU_01639_akt.zip",
    "stundenwerte_TU_01645_akt.zip",
    "stundenwerte_TU_01666_akt.zip",
    "stundenwerte_TU_01684_akt.zip",
    "stundenwerte_TU_01691_akt.zip",
    "stundenwerte_TU_01694_akt.zip",
    "stundenwerte_TU_01721_akt.zip",
    "stundenwerte_TU_01735_akt.zip",
    "stundenwerte_TU_01736_akt.zip",
    "stundenwerte_TU_01757_akt.zip",
    "stundenwerte_TU_01759_akt.zip",
    "stundenwerte_TU_01766_akt.zip",
    "stundenwerte_TU_01792_akt.zip",
    "stundenwerte_TU_01803_akt.zip",
    "stundenwerte_TU_01832_akt.zip",
    "stundenwerte_TU_01863_akt.zip",
    "stundenwerte_TU_01869_akt.zip",
    "stundenwerte_TU_01886_akt.zip",
    "stundenwerte_TU_01964_akt.zip",
    "stundenwerte_TU_01975_akt.zip",
    "stundenwerte_TU_01981_akt.zip",
    "stundenwerte_TU_02014_akt.zip",
    "stundenwerte_TU_02023_akt.zip",
    "stundenwerte_TU_02039_akt.zip",
    "stundenwerte_TU_02044_akt.zip",
    "stundenwerte_TU_02074_akt.zip",
    "stundenwerte_TU_02110_akt.zip",
    "stundenwerte_TU_02115_akt.zip",
    "stundenwerte_TU_02171_akt.zip",
    "stundenwerte_TU_02174_akt.zip",
    "stundenwerte_TU_02201_akt.zip",
    "stundenwerte_TU_02211_akt.zip",
    "stundenwerte_TU_02252_akt.zip",
    "stundenwerte_TU_02261_akt.zip",
    "stundenwerte_TU_02290_akt.zip",
    "stundenwerte_TU_02303_akt.zip",
    "stundenwerte_TU_02306_akt.zip",
    "stundenwerte_TU_02315_akt.zip",
    "stundenwerte_TU_02319_akt.zip",
    "stundenwerte_TU_02323_akt.zip",
    "stundenwerte_TU_02362_akt.zip",
    "stundenwerte_TU_02385_akt.zip",
    "stundenwerte_TU_02410_akt.zip",
    "stundenwerte_TU_02429_akt.zip",
    "stundenwerte_TU_02437_akt.zip",
    "stundenwerte_TU_02444_akt.zip",
    "stundenwerte_TU_02480_akt.zip",
    "stundenwerte_TU_02483_akt.zip",
    "stundenwerte_TU_02485_akt.zip",
    "stundenwerte_TU_02486_akt.zip",
    "stundenwerte_TU_02497_akt.zip",
    "stundenwerte_TU_02559_akt.zip",
    "stundenwerte_TU_02564_akt.zip",
    "stundenwerte_TU_02575_akt.zip",
    "stundenwerte_TU_02578_akt.zip",
    "stundenwerte_TU_02597_akt.zip",
    "stundenwerte_TU_02600_akt.zip",
    "stundenwerte_TU_02601_akt.zip",
    "stundenwerte_TU_02618_akt.zip",
    "stundenwerte_TU_02627_akt.zip",
    "stundenwerte_TU_02629_akt.zip",
    "stundenwerte_TU_02638_akt.zip",
    "stundenwerte_TU_02641_akt.zip",
    "stundenwerte_TU_02667_akt.zip",
    "stundenwerte_TU_02680_akt.zip",
    "stundenwerte_TU_02700_akt.zip",
    "stundenwerte_TU_02704_akt.zip",
    "stundenwerte_TU_02708_akt.zip",
    "stundenwerte_TU_02712_akt.zip",
    "stundenwerte_TU_02750_akt.zip",
    "stundenwerte_TU_02773_akt.zip",
    "stundenwerte_TU_02794_akt.zip",
    "stundenwerte_TU_02796_akt.zip",
    "stundenwerte_TU_02812_akt.zip",
    "stundenwerte_TU_02814_akt.zip",
    "stundenwerte_TU_02856_akt.zip",
    "stundenwerte_TU_02878_akt.zip",
    "stundenwerte_TU_02886_akt.zip",
    "stundenwerte_TU_02905_akt.zip",
    "stundenwerte_TU_02907_akt.zip",
    "stundenwerte_TU_02925_akt.zip",
    "stundenwerte_TU_02928_akt.zip",
    "stundenwerte_TU_02932_akt.zip",
    "stundenwerte_TU_02947_akt.zip",
    "stundenwerte_TU_02951_akt.zip",
    "stundenwerte_TU_02953_akt.zip",
    "stundenwerte_TU_02961_akt.zip",
    "stundenwerte_TU_02968_akt.zip",
    "stundenwerte_TU_02985_akt.zip",
    "stundenwerte_TU_03015_akt.zip",
    "stundenwerte_TU_03028_akt.zip",
    "stundenwerte_TU_03031_akt.zip",
    "stundenwerte_TU_03032_akt.zip",
    "stundenwerte_TU_03034_akt.zip",
    "stundenwerte_TU_03042_akt.zip",
    "stundenwerte_TU_03083_akt.zip",
    "stundenwerte_TU_03086_akt.zip",
    "stundenwerte_TU_03093_akt.zip",
    "stundenwerte_TU_03098_akt.zip",
    "stundenwerte_TU_03126_akt.zip",
    "stundenwerte_TU_03137_akt.zip",
    "stundenwerte_TU_03147_akt.zip",
    "stundenwerte_TU_03155_akt.zip",
    "stundenwerte_TU_03158_akt.zip",
    "stundenwerte_TU_03164_akt.zip",
    "stundenwerte_TU_03166_akt.zip",
    "stundenwerte_TU_03167_akt.zip",
    "stundenwerte_TU_03181_akt.zip",
    "stundenwerte_TU_03196_akt.zip",
    "stundenwerte_TU_03204_akt.zip",
    "stundenwerte_TU_03226_akt.zip",
    "stundenwerte_TU_03231_akt.zip",
    "stundenwerte_TU_03234_akt.zip",
    "stundenwerte_TU_03244_akt.zip",
    "stundenwerte_TU_03254_akt.zip",
    "stundenwerte_TU_03257_akt.zip",
    "stundenwerte_TU_03268_akt.zip",
    "stundenwerte_TU_03271_akt.zip",
    "stundenwerte_TU_03278_akt.zip",
    "stundenwerte_TU_03284_akt.zip",
    "stundenwerte_TU_03287_akt.zip",
    "stundenwerte_TU_03289_akt.zip",
    "stundenwerte_TU_03307_akt.zip",
    "stundenwerte_TU_03319_akt.zip",
    "stundenwerte_TU_03321_akt.zip",
    "stundenwerte_TU_03340_akt.zip",
    "stundenwerte_TU_03348_akt.zip",
    "stundenwerte_TU_03362_akt.zip",
    "stundenwerte_TU_03366_akt.zip",
    "stundenwerte_TU_03376_akt.zip",
    "stundenwerte_TU_03379_akt.zip",
    "stundenwerte_TU_03402_akt.zip",
    "stundenwerte_TU_03426_akt.zip",
    "stundenwerte_TU_03442_akt.zip",
    "stundenwerte_TU_03484_akt.zip",
    "stundenwerte_TU_03485_akt.zip",
    "stundenwerte_TU_03490_akt.zip",
    "stundenwerte_TU_03509_akt.zip",
    "stundenwerte_TU_03513_akt.zip",
    "stundenwerte_TU_03527_akt.zip",
    "stundenwerte_TU_03540_akt.zip",
    "stundenwerte_TU_03545_akt.zip",
    "stundenwerte_TU_03571_akt.zip",
    "stundenwerte_TU_03591_akt.zip",
    "stundenwerte_TU_03603_akt.zip",
    "stundenwerte_TU_03612_akt.zip",
    "stundenwerte_TU_03621_akt.zip",
    "stundenwerte_TU_03623_akt.zip",
    "stundenwerte_TU_03631_akt.zip",
    "stundenwerte_TU_03639_akt.zip",
    "stundenwerte_TU_03660_akt.zip",
    "stundenwerte_TU_03667_akt.zip",
    "stundenwerte_TU_03668_akt.zip",
    "stundenwerte_TU_03679_akt.zip",
    "stundenwerte_TU_03730_akt.zip",
    "stundenwerte_TU_03734_akt.zip",
    "stundenwerte_TU_03739_akt.zip",
    "stundenwerte_TU_03761_akt.zip",
    "stundenwerte_TU_03811_akt.zip",
    "stundenwerte_TU_03821_akt.zip",
    "stundenwerte_TU_03836_akt.zip",
    "stundenwerte_TU_03857_akt.zip",
    "stundenwerte_TU_03875_akt.zip",
    "stundenwerte_TU_03897_akt.zip",
    "stundenwerte_TU_03904_akt.zip",
    "stundenwerte_TU_03925_akt.zip",
    "stundenwerte_TU_03927_akt.zip",
    "stundenwerte_TU_03939_akt.zip",
    "stundenwerte_TU_03946_akt.zip",
    "stundenwerte_TU_03975_akt.zip",
    "stundenwerte_TU_03987_akt.zip",
    "stundenwerte_TU_04024_akt.zip",
    "stundenwerte_TU_04032_akt.zip",
    "stundenwerte_TU_04036_akt.zip",
    "stundenwerte_TU_04039_akt.zip",
    "stundenwerte_TU_04063_akt.zip",
    "stundenwerte_TU_04094_akt.zip",
    "stundenwerte_TU_04104_akt.zip",
    "stundenwerte_TU_04127_akt.zip",
    "stundenwerte_TU_04160_akt.zip",
    "stundenwerte_TU_04169_akt.zip",
    "stundenwerte_TU_04175_akt.zip",
    "stundenwerte_TU_04177_akt.zip",
    "stundenwerte_TU_04189_akt.zip",
    "stundenwerte_TU_04261_akt.zip",
    "stundenwerte_TU_04271_akt.zip",
    "stundenwerte_TU_04275_akt.zip",
    "stundenwerte_TU_04280_akt.zip",
    "stundenwerte_TU_04287_akt.zip",
    "stundenwerte_TU_04300_akt.zip",
    "stundenwerte_TU_04301_akt.zip",
    "stundenwerte_TU_04323_akt.zip",
    "stundenwerte_TU_04336_akt.zip",
    "stundenwerte_TU_04349_akt.zip",
    "stundenwerte_TU_04354_akt.zip",
    "stundenwerte_TU_04371_akt.zip",
    "stundenwerte_TU_04377_akt.zip",
    "stundenwerte_TU_04393_akt.zip",
    "stundenwerte_TU_04411_akt.zip",
    "stundenwerte_TU_04445_akt.zip",
    "stundenwerte_TU_04464_akt.zip",
    "stundenwerte_TU_04466_akt.zip",
    "stundenwerte_TU_04480_akt.zip",
    "stundenwerte_TU_04501_akt.zip",
    "stundenwerte_TU_04508_akt.zip",
    "stundenwerte_TU_04548_akt.zip",
    "stundenwerte_TU_04559_akt.zip",
    "stundenwerte_TU_04560_akt.zip",
    "stundenwerte_TU_04592_akt.zip",
    "stundenwerte_TU_04605_akt.zip",
    "stundenwerte_TU_04625_akt.zip",
    "stundenwerte_TU_04642_akt.zip",
    "stundenwerte_TU_04651_akt.zip",
    "stundenwerte_TU_04703_akt.zip",
    "stundenwerte_TU_04704_akt.zip",
    "stundenwerte_TU_04706_akt.zip",
    "stundenwerte_TU_04709_akt.zip",
    "stundenwerte_TU_04745_akt.zip",
    "stundenwerte_TU_04748_akt.zip",
    "stundenwerte_TU_04763_akt.zip",
    "stundenwerte_TU_04813_akt.zip",
    "stundenwerte_TU_04841_akt.zip",
    "stundenwerte_TU_04857_akt.zip",
    "stundenwerte_TU_04878_akt.zip",
    "stundenwerte_TU_04887_akt.zip",
    "stundenwerte_TU_04896_akt.zip",
    "stundenwerte_TU_04911_akt.zip",
    "stundenwerte_TU_04928_akt.zip",
    "stundenwerte_TU_04931_akt.zip",
    "stundenwerte_TU_04978_akt.zip",
    "stundenwerte_TU_04997_akt.zip",
    "stundenwerte_TU_05009_akt.zip",
    "stundenwerte_TU_05014_akt.zip",
    "stundenwerte_TU_05017_akt.zip",
    "stundenwerte_TU_05029_akt.zip",
    "stundenwerte_TU_05046_akt.zip",
    "stundenwerte_TU_05064_akt.zip",
    "stundenwerte_TU_05097_akt.zip",
    "stundenwerte_TU_05099_akt.zip",
    "stundenwerte_TU_05100_akt.zip",
    "stundenwerte_TU_05109_akt.zip",
    "stundenwerte_TU_05111_akt.zip",
    "stundenwerte_TU_05133_akt.zip",
    "stundenwerte_TU_05142_akt.zip",
    "stundenwerte_TU_05146_akt.zip",
    "stundenwerte_TU_05149_akt.zip",
    "stundenwerte_TU_05158_akt.zip",
    "stundenwerte_TU_05229_akt.zip",
    "stundenwerte_TU_05275_akt.zip",
    "stundenwerte_TU_05279_akt.zip",
    "stundenwerte_TU_05280_akt.zip",
    "stundenwerte_TU_05300_akt.zip",
    "stundenwerte_TU_05335_akt.zip",
    "stundenwerte_TU_05347_akt.zip",
    "stundenwerte_TU_05349_akt.zip",
    "stundenwerte_TU_05371_akt.zip",
    "stundenwerte_TU_05397_akt.zip",
    "stundenwerte_TU_05404_akt.zip",
    "stundenwerte_TU_05424_akt.zip",
    "stundenwerte_TU_05426_akt.zip",
    "stundenwerte_TU_05433_akt.zip",
    "stundenwerte_TU_05440_akt.zip",
    "stundenwerte_TU_05480_akt.zip",
    "stundenwerte_TU_05490_akt.zip",
    "stundenwerte_TU_05516_akt.zip",
    "stundenwerte_TU_05538_akt.zip",
    "stundenwerte_TU_05541_akt.zip",
    "stundenwerte_TU_05546_akt.zip",
    "stundenwerte_TU_05562_akt.zip",
    "stundenwerte_TU_05629_akt.zip",
    "stundenwerte_TU_05640_akt.zip",
    "stundenwerte_TU_05643_akt.zip",
    "stundenwerte_TU_05664_akt.zip",
    "stundenwerte_TU_05676_akt.zip",
    "stundenwerte_TU_05688_akt.zip",
    "stundenwerte_TU_05692_akt.zip",
    "stundenwerte_TU_05705_akt.zip",
    "stundenwerte_TU_05715_akt.zip",
    "stundenwerte_TU_05717_akt.zip",
    "stundenwerte_TU_05731_akt.zip",
    "stundenwerte_TU_05745_akt.zip",
    "stundenwerte_TU_05750_akt.zip",
    "stundenwerte_TU_05779_akt.zip",
    "stundenwerte_TU_05792_akt.zip",
    "stundenwerte_TU_05797_akt.zip",
    "stundenwerte_TU_05800_akt.zip",
    "stundenwerte_TU_05822_akt.zip",
    "stundenwerte_TU_05825_akt.zip",
    "stundenwerte_TU_05839_akt.zip",
    "stundenwerte_TU_05856_akt.zip",
    "stundenwerte_TU_05871_akt.zip",
    "stundenwerte_TU_05906_akt.zip",
    "stundenwerte_TU_05930_akt.zip",
    "stundenwerte_TU_05941_akt.zip",
    "stundenwerte_TU_06093_akt.zip",
    "stundenwerte_TU_06105_akt.zip",
    "stundenwerte_TU_06109_akt.zip",
    "stundenwerte_TU_06129_akt.zip",
    "stundenwerte_TU_06157_akt.zip",
    "stundenwerte_TU_06158_akt.zip",
    "stundenwerte_TU_06159_akt.zip",
    "stundenwerte_TU_06163_akt.zip",
    "stundenwerte_TU_06170_akt.zip",
    "stundenwerte_TU_06197_akt.zip",
    "stundenwerte_TU_06199_akt.zip",
    "stundenwerte_TU_06217_akt.zip",
    "stundenwerte_TU_06258_akt.zip",
    "stundenwerte_TU_06259_akt.zip",
    "stundenwerte_TU_06260_akt.zip",
    "stundenwerte_TU_06262_akt.zip",
    "stundenwerte_TU_06263_akt.zip",
    "stundenwerte_TU_06264_akt.zip",
    "stundenwerte_TU_06265_akt.zip",
    "stundenwerte_TU_06266_akt.zip",
    "stundenwerte_TU_06272_akt.zip",
    "stundenwerte_TU_06273_akt.zip",
    "stundenwerte_TU_06275_akt.zip",
    "stundenwerte_TU_06305_akt.zip",
    "stundenwerte_TU_06310_akt.zip",
    "stundenwerte_TU_06314_akt.zip",
    "stundenwerte_TU_06336_akt.zip",
    "stundenwerte_TU_06337_akt.zip",
    "stundenwerte_TU_06344_akt.zip",
    "stundenwerte_TU_06346_akt.zip",
    "stundenwerte_TU_06347_akt.zip",
    "stundenwerte_TU_07075_akt.zip",
    "stundenwerte_TU_07099_akt.zip",
    "stundenwerte_TU_07105_akt.zip",
    "stundenwerte_TU_07106_akt.zip",
    "stundenwerte_TU_07187_akt.zip",
    "stundenwerte_TU_07298_akt.zip",
    "stundenwerte_TU_07319_akt.zip",
    "stundenwerte_TU_07321_akt.zip",
    "stundenwerte_TU_07329_akt.zip",
    "stundenwerte_TU_07330_akt.zip",
    "stundenwerte_TU_07331_akt.zip",
    "stundenwerte_TU_07341_akt.zip",
    "stundenwerte_TU_07343_akt.zip",
    "stundenwerte_TU_07350_akt.zip",
    "stundenwerte_TU_07351_akt.zip",
    "stundenwerte_TU_07364_akt.zip",
    "stundenwerte_TU_07367_akt.zip",
    "stundenwerte_TU_07368_akt.zip",
    "stundenwerte_TU_07369_akt.zip",
    "stundenwerte_TU_07370_akt.zip",
    "stundenwerte_TU_07373_akt.zip",
    "stundenwerte_TU_07374_akt.zip",
    "stundenwerte_TU_07389_akt.zip",
    "stundenwerte_TU_07393_akt.zip",
    "stundenwerte_TU_07394_akt.zip",
    "stundenwerte_TU_07395_akt.zip",
    "stundenwerte_TU_07396_akt.zip",
    "stundenwerte_TU_07403_akt.zip",
    "stundenwerte_TU_07410_akt.zip",
    "stundenwerte_TU_07412_akt.zip",
    "stundenwerte_TU_07419_akt.zip",
    "stundenwerte_TU_07420_akt.zip",
    "stundenwerte_TU_07424_akt.zip",
    "stundenwerte_TU_07427_akt.zip",
    "stundenwerte_TU_07428_akt.zip",
    "stundenwerte_TU_07431_akt.zip",
    "stundenwerte_TU_07432_akt.zip",
    "stundenwerte_TU_13670_akt.zip",
    "stundenwerte_TU_13674_akt.zip",
    "stundenwerte_TU_13675_akt.zip",
    "stundenwerte_TU_13696_akt.zip",
    "stundenwerte_TU_13700_akt.zip",
    "stundenwerte_TU_13710_akt.zip",
    "stundenwerte_TU_13711_akt.zip",
    "stundenwerte_TU_13713_akt.zip",
    "stundenwerte_TU_13777_akt.zip",
    "stundenwerte_TU_13965_akt.zip",
    "stundenwerte_TU_15000_akt.zip",
    "stundenwerte_TU_15207_akt.zip",
    "stundenwerte_TU_15444_akt.zip",
    "stundenwerte_TU_15555_akt.zip",
    "stundenwerte_TU_15813_akt.zip",
    "stundenwerte_TU_19171_akt.zip",
    "stundenwerte_TU_19172_akt.zip",
    "stundenwerte_TU_19207_akt.zip",
]

from_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent"

zipped_dir = f"{paths.root}/data/raw/dwd/airtemp2m/zipped"
unzipped_dir = paths.raw_dwd


def download_dwd():
    os.makedirs(zipped_dir, exist_ok=True)
    os.makedirs(unzipped_dir, exist_ok=True)

    print("Downloading DWD data")
    for fname in tqdm(fnames):
        url = f"{from_url}/{fname}"

        if fname.endswith("zip"):
            out_fpath = f"{zipped_dir}/{fname}"
        else:
            out_fpath = f"{unzipped_dir}/{fname}"
        if os.path.exists(out_fpath):
            continue
        urllib.request.urlretrieve(url, out_fpath)

    print("Unzipping DWD data")
    for fname in tqdm(fnames):
        if not fname.endswith("zip"):
            continue

        zip_fpath = f"{zipped_dir}/{fname}"

        fname_no_ext = fname.split(".")[0]
        out_dir = f"{unzipped_dir}/{fname_no_ext}"
        os.makedirs(out_dir, exist_ok=True)

        try:
            with ZipFile(zip_fpath, "r") as f:
                f.extractall(out_dir)
        except BadZipFile:
            print(
                f"WARNING: {fname} is corrupted. Please delete it and restart the process."
            )


def load_station_df(fpath):
    df = pd.read_csv(fpath, delimiter=";", encoding="latin-1")
    df.columns = [names.station_id, names.time, "QN_9", names.temp, "RF_TU", "eor"]
    # drop relative humidity, "end of record" column.
    df = df.drop(["RF_TU", "eor", "QN_9"], axis=1)
    # Filter date.
    df[names.time] = pd.to_datetime(df[names.time], format="%Y%m%d%H")

    # Filter invalid values.
    df = df[df[names.temp] != -999.0]
    return df


def load_station_metadata(fpath):
    meta_columns = [
        names.station_id,
        names.height,
        names.lat,
        names.lon,
        "FROM_DATE",
        "TO_DATE",
        names.station_name,
    ]
    df = pd.read_csv(fpath, delimiter=";", encoding="latin-1")
    df.columns = meta_columns

    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"], format="%Y%m%d")

    # Last "TO_DATE" is empty because it represents current location.
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"], format="%Y%m%d", errors="coerce")
    df.loc[df.index[-1], "TO_DATE"] = pd.to_datetime("today").normalize()
    return df


def process_dwd():
    dfs = []
    meta_dfs = []
    root = paths.raw_dwd
    print("Processing DWD data")
    for subdir in tqdm(os.listdir(root)):
        dir_path = f"{root}/{subdir}"
        if not os.path.isdir(dir_path):
            continue

        # Data:
        fname = [n for n in os.listdir(dir_path) if n.startswith("produkt")][0]
        fpath = f"{root}/{subdir}/{fname}"
        dfs.append(load_station_df(fpath))

        # Metadata:
        fname = [n for n in os.listdir(dir_path) if n.startswith("Metadaten_Geo")][0]
        fpath = f"{root}/{subdir}/{fname}"
        meta_dfs.append(load_station_metadata(fpath))

    df = pd.concat(dfs, ignore_index=True)
    meta_df = pd.concat(meta_dfs, ignore_index=True)
    geometry = gpd.points_from_xy(meta_df[names.lon], meta_df[names.lat])
    meta_df = gpd.GeoDataFrame(meta_df, geometry=geometry)
    meta_df.crs = data.crs_str

    # Filter days with not enoungh data. (Mainly at the start of dataset period).
    counts = df.set_index(names.time).groupby(names.time).count()
    good_times = counts[counts[names.station_id] > 400].index.unique()
    df = df[df[names.time].isin(good_times)].reset_index(drop=True)

    # cache for faster loading in future.
    ensure_dir_exists(paths.dwd)
    ensure_dir_exists(paths.dwd_meta)
    df.to_feather(paths.dwd)
    meta_df.to_feather(paths.dwd_meta)


def process_value_stations():
    df = pd.read_csv(VALUE_STATIONS_FPATH)
    df.columns = [
        names.station_id,
        names.station_name,
        names.lon,
        names.lat,
        names.height,
        "source",
    ]

    # StationID does not match with DWD dataset.
    df = df.drop([names.station_id, "source"], axis=1)

    # Strip whitespace:
    df[names.station_name] = df[names.station_name].str.strip()

    geometry = gpd.points_from_xy(df[names.lon], df[names.lat])
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df.crs = data.crs_str
    df.to_feather(paths.dwd_test_stations)


def datetime_split_plot():
    from sim2real.plots import save_plot
    import matplotlib.pyplot as plt

    full = DWDStationData(paths).full()
    dts = full.index.get_level_values(names.time).unique()

    train, val, test = train_val_test_dts(dts)
    plt.figure(figsize=(6, 1))
    plt.scatter(train, [1 for _ in train], s=100, marker="|", label="Train")
    plt.scatter(val, [1 for _ in val], s=100, marker="|", label="Val")
    plt.scatter(test, [1 for _ in test], s=100, marker="|", label="Test")

    plt.xlim(pd.Timestamp("2021-11-03"), pd.Timestamp("2022-04-01"))
    plt.ylim(0.5, 2)
    plt.legend(ncol=3)
    plt.xticks(rotation=45)
    plt.yticks([])
    save_plot(None, "datetime_split")


def train_val_test_dts(dts):
    """
    This follows the google paper's sampling strategy.
    """
    dts = list(dts)

    train, val, test = [], [], []

    # 19 days.
    train_duration = 19 * 24
    # 2 days.
    val_duration = 2 * 24
    # 2.5 days.
    test_duration = 5 * 12

    # 2 days skipped at borders.
    skip_duration = 2 * 24

    i = 0
    while i < len(dts):
        # Add training datetimes.
        train += dts[i : i + train_duration]
        i += train_duration + skip_duration
        if i >= len(dts):
            break

        val += dts[i : i + val_duration]
        i += val_duration + skip_duration
        if i >= len(dts):
            break

        test += dts[i : i + test_duration]
        i += test_duration + skip_duration
        if i >= len(dts):
            break

    # Make sure there's no overlap.
    assert (
        set(train).isdisjoint(val)
        and set(val).isdisjoint(test)
        and set(test).isdisjoint(train)
    )

    return train, val, test


def split(df, dts, station_ids) -> Tuple[pd.DataFrame]:
    """
    Split a dataframe by BOTH datetimes and station ids.

    Returns: (split, remainder): pd.DataFrame
    """

    split = df.query(f"{names.station_id} in @station_ids and {names.time} in @dts")
    remainder = df.query(
        f"{names.station_id} not in @station_ids and {names.time} not in @dts"
    )

    return split, remainder


def get_test_station_ids():
    dwd = DWDStationData(paths)

    df = pd.read_csv(VALUE_STATIONS_FPATH)
    df.columns = [
        names.station_id,
        names.station_name,
        names.lon,
        names.lat,
        names.height,
        "source",
    ]

    # StationID does not match with DWD dataset.
    df = df.drop([names.station_id, "source"], axis=1)

    # Strip whitespace:
    df[names.station_name] = df[names.station_name].str.strip()

    geometry = gpd.points_from_xy(df[names.lon], df[names.lat])
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df.crs = data.crs_str

    test_station_ids = set(df.sjoin_nearest(dwd.meta_df)[names.station_id])
    return test_station_ids


def distance_matrix(gdf1, gdf2):
    # Station distance matrix:
    return gdf1.geometry.apply(lambda g: gdf2.distance(g))


def pick_stations(stations, distance_matrix, num_stations_left):
    if len(stations) == 0:
        # Start somewhere.
        stations = []
        num_stations_left -= 1

    if num_stations_left == 0:
        return stations

    stations.append(get_furthest(stations, distance_matrix))
    return pick_stations(stations, distance_matrix, num_stations_left - 1)


def get_furthest(stations, distance_matrix):
    """
    Gets the furthest station from all the current stations.
    """
    smallest = distance_matrix[stations].min(axis=1)
    return smallest[smallest == smallest.max()].index[0]


def station_sampling_plot():
    full = DWDStationData(paths).full()
    gdf = full.groupby(names.station_id).first()

    from sim2real.plots import init_fig, save_plot

    fig, axs, transform = init_fig(1, 4, ret_transform=True, figsize=(14, 6))

    Ns = [1, 5, 50, 200]

    for N, ax in zip(Ns, axs):
        stations = pick_stations([13777], distance_matrix(gdf, gdf), N - 1)
        subset = gdf.query(f"{names.station_id} in @stations")
        subset.plot(
            ax=ax, transform=transform, marker="o", facecolor="none", color="C0"
        )
        ax.set_title(f"$N = {N}$")

    save_plot(None, "station_sampling", fig)


def save_station_splits(mode="random"):
    num_val_stations = 50
    np.random.seed(42)

    dwd = DWDStationData(paths)
    gdf = dwd.full().groupby(names.station_id).first()
    station_ids = gdf.index.get_level_values(names.station_id).sort_values()
    sdf = pd.DataFrame(index=station_ids)
    sdf["SET"] = None
    sdf["ORDER"] = 0

    test_station_ids = get_test_station_ids()

    # Define test stations.
    sdf.loc[list(test_station_ids), "SET"] = "TEST"
    # Remove test stations from the pool.
    station_ids = set(station_ids) - set(test_station_ids)

    # Define validation stations.
    gdf = gdf.query(f"{names.station_id} in @station_ids")

    if mode == "random":
        val_station_ids = np.random.choice(list(station_ids), num_val_stations)
    else:
        dm = distance_matrix(gdf, gdf)
        val_station_ids = pick_stations(
            [next(iter(station_ids))], dm, num_val_stations - 1
        )

    sdf.loc[val_station_ids, "SET"] = "VAL"
    sdf.loc[val_station_ids, "ORDER"] = list(range(len(val_station_ids)))

    # Remove val stations from the pool.
    station_ids = station_ids - set(val_station_ids)
    gdf = gdf.query(f"{names.station_id} in @station_ids")

    if mode == "random":
        train_station_ids = list(station_ids)
        np.random.shuffle(train_station_ids)
    else:
        # Define train stations.
        dm = distance_matrix(gdf, gdf)
        num_train_stations = len(station_ids)
        train_station_ids = pick_stations(
            [next(iter(station_ids))], dm, num_train_stations - 1
        )
    sdf.loc[train_station_ids, "SET"] = "TRAIN"
    sdf.loc[train_station_ids, "ORDER"] = list(range(len(train_station_ids)))
    sdf = sdf.reset_index()

    ensure_dir_exists(paths.station_split)
    sdf.to_feather(paths.station_split)


def save_datetime_splits():
    full = DWDStationData(paths).full()
    dts = full.index.get_level_values(names.time).unique()
    train_dts, val_dts, test_dts = train_val_test_dts(dts)
    df = pd.DataFrame(index=dts.sort_values())
    df["SET"] = None
    df.loc[train_dts] = "TRAIN"
    df.loc[val_dts] = "VAL"
    df.loc[test_dts] = "TEST"
    df = df.reset_index()
    ensure_dir_exists(paths.time_split)
    df.to_feather(paths.time_split)


def plot_train_val(train, val):
    fig, axs = init_fig()
    dwd = DWDStationData(paths)
    dwd.plot_stations(train, "o", "C0", ax=axs[0], label="Train")
    dwd.plot_stations(val, "s", "C1", ax=axs[0], label="Val")
    axs[0].legend()


if __name__ == "__main__":
    download_dwd()
    process_dwd()
    process_value_stations()
    datetime_split_plot()
    save_station_splits("random")
    save_datetime_splits()
    ss = load_station_splits()
