# -*- coding: utf-8 -*-
import os
import codecs
import re


def set_tag_bio(text,tag):
    txt = ""
    if tag == "O":
        # segment = jieba.posseg.cut(text)
        for i in text:
            x = i + "\t" + "O" + "\n"
            if (len(x.split()) >= 2):
                txt += i + "\t" + "O" + "\n"

    else:
        # segment = list(jieba.posseg.cut(text))
        txt += text[0] + "\t" + "B-" + tag[0:3] + "\n"
        for i in range(1, len(text)):
            txt += text[i] + "\t" + "I-" + tag[0:3] + "\n"
    return txt
    # txt = ""
    # if tag == "O":
    #     segment = jieba.posseg.cut(text)
    #     for seg in segment:
    #         txt += seg.word + "\t" + seg.flag + "\t" + "O" + "\n"
    # else:
    #     segment = list(jieba.posseg.cut(text))
    #     txt += segment[0].word + "\t" + segment[0].flag + "\t" + "B-" + tag + "\n"
    #     for i in range(1, len(segment)):
    #         txt += segment[i].word + "\t" + segment[i].flag + "\t" + "I-" + tag + "\n"
    # return txt

def tag_txt_ner():
    text = ""
    str = "泳儿（Vincy Chan），新加坡女[[[MUI&&&歌手]]]。1982年10月16日生于中国香港，1990年移民新加坡。2005年毕业于新加坡南洋理工大学会计系，同年回香港参加“2005年度英皇新秀歌唱大赛”，夺得亚军及最佳型格奖。2006年加盟[[[ORG&&&英皇娱乐集团有限公司]]]，晋身乐坛。2006年，举行首个个人音乐会，并夺得2006年度十大中文金曲“最有前途新人奖金奖”、[[[MUI&&&十大劲歌金曲]]]颁奖典礼“最受欢迎新人奖金奖、叱吒乐坛流行榜颁奖礼“叱吒乐坛生力军金奖”和新城劲爆颁奖礼“新城劲爆女新人王”。2007年，单曲《黛玉笑了》成为她的首支四台冠军歌。2013年，出演音乐舞台剧《天狐变》。2014年，发行国语HiFi专辑《爱·情歌》，6月参加安徽卫视歌唱节目《我为歌狂》，12月夺广东电视台《[[[MOV&&&麦王争霸]]]》冠军。\n2005年，泳儿新加坡南洋理工大学会计系毕业之后，回港参加“2005年度英皇新秀歌唱大赛”，夺得亚军及最佳型格奖。2006年加盟[[[ORG&&&英皇娱乐集团有限公司]]]Music Plus，晋身乐坛。2006年，泳儿发行首张个人大碟《感应》，同名主打歌《感应》是台湾女歌手[[[PER&&&林凡]]]《一个人生活》的广东话翻唱版本，荣登各大电台及电视台榜首名，成为三台冠军歌，荣获第29届十大中文金曲的金曲奖。2006年8月20日，泳儿举行首个个人音乐会《感应泳儿音乐会》。年底，泳儿获得2006年度叱吒乐坛流行榜颁奖礼“叱吒乐坛生力军金奖”、新城劲爆颁奖礼“新城劲爆女新人王”、十大中文金曲 “最有前途新人奖金奖”和[[[MUI&&&十大劲歌金曲]]]颁奖典礼“最受欢迎新人奖金奖”，以及加拿大Hit中文歌曲排行榜“全国推崇新女歌手奖”。《爱·情歌》2007年，泳儿推出的第二张个人专辑《花无雪》，主打歌《黛玉笑了》成为香港电台、903、香港新城电台和TVB劲歌金曲的四台冠军歌，《无心恋唱》获得十大劲歌金曲颁奖典礼的金曲奖。同年11月27日，泳儿于九龙湾国际展贸中心Star Hall举行《泳儿Close To You Live音乐会》。2008年5月31日，泳儿在马来西亚云顶举行首场个人海外音乐会《感应泳儿云顶音乐会》。6月，泳儿推出国语专辑《多想认识你》，国语歌曲《喜欢一个人好累》荣获第三十一届十大中文金曲“优秀流行国语歌曲奖” 。"
    strlist = re.split(u"\[\[\[|\]\]\]",str)
    for txt in strlist:
        if len(re.findall(u"&&&",txt)):
            text += set_tag_bio(txt.split("&&&")[1],txt.split("&&&")[0])
        else:
            text += set_tag_bio(txt, "O")
    print(text)
    return text


def product_tag(path,ltp):
    f_list = os.listdir(path)
    testfile = codecs.open("./NER/music_test.txt","a","utf-8")
    trainfile = codecs.open("./NER/music_train.txt","a","utf-8")
    train_file_num = 430
    file_num = 0
    for file in f_list:
        if os.path.splitext(file)[1] == '.ann':
            ner_list = get_ann_data(path + '/' + file)
            text = tag_txt_ner(path + '/' + os.path.splitext(file)[0] + '.txt', ner_list, ltp)
            if file_num < train_file_num:
                pass
                trainfile.write(text)
            else:
                testfile.write(text)
            file_num += 1
tag_txt_ner()

#product_tag("./data",ltp)
# delete_null_files("./data")
# get_all_files("./data")