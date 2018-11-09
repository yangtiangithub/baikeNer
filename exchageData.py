import config
import pymongo
import re
import redis
import sys
import jieba
import jieba.posseg
import codecs
import multiprocessing

class Ner():
    """
    实体标注类，主要用来构造实体识别标注数据集

    属性：
        mongodb:初始化mongodb连接
        urlRedis:初始化redis连接
    """
    def __init__(self):
        client = pymongo.MongoClient(host=config.MONGODB_SERVER,port=config.MONGODB_PORT)
        self.mongodb = client[config.MONGODB_DB]
        self.urlRedis = redis.StrictRedis(host=config.REDIS_SEVER, port=config.REDIS_PORT, db=2)

    def exchange_mongo_data(self):
        """
        根据爬取到的百科文本构造实体弱标注数据集，主要包含的实体类有：机构，人物，音乐，专辑，影视
        :return: 无
        """
        collection = self.mongodb.newBaike
        # cursor = collection.find({"$or":[{"演艺经历": {"$ne": ""}}, {"summary": {"$ne": ""}}, {"早年经历": {"$ne": ""}}]})
        # 设置batch_size避免超时，或者设置成find({},no_cursor_timeout = True)
        cursor = collection.find({"演艺经历": {"$ne": None}}).batch_size(50)
        for person in cursor:
            text = ""
            if "summary" in person.keys():
                text += person["summary"] + "\n"
            if "早年经历".decode("utf-8") in person.keys():
                text += person["早年经历".decode("utf-8")] + "\n"
            if "演艺经历".decode("utf-8") in person.keys():
                text += person["演艺经历".decode("utf-8")]
            url = person["url"]
            self.url_to_tag(url,text)
    def url_to_tag(self,url,text):
        """
        从mongodb中查询括号里面实体的标签，并替代它的url，如果mongodb里面查询不到则去掉括号和url，
        仅仅保留其文本
        :param url: 这个百科文本的realUrl
        :param text: 这个百科文本的summary + 早年经历 + 演艺经历组成的文本
        :return:无
        """
        try:
            collection = self.mongodb.newBaike
            url_list = re.findall(u'\[\[\[(.*?)\]\]\]',text)
            if len(url_list) > 0:
                for url_name in url_list:
                    # 获取realUrl
                    realUrl = self.urlRedis.smembers(url_name.split("&&&")[0])
                    if len(realUrl):
                        object = collection.find_one({"url":realUrl.pop()})
                        if object != None:
                            tag = object["tag"]
                            if len(re.findall(u"公司|学校",tag)):
                                text = text.replace(url_name.split("&&&")[0], "ORG")
                            elif len(re.findall(u"歌手|演员|人物",tag)):
                                text = text.replace(url_name.split("&&&")[0], "PER")
                            elif len(re.findall(u"单曲|音乐", tag)):
                                text = text.replace(url_name.split("&&&")[0], "MUI")
                            elif len(re.findall(u"专辑",tag)):
                                text = text.replace(url_name.split("&&&")[0], "ALB")
                            elif len(re.findall(u"电影|电视剧",tag)):
                                text = text.replace(url_name.split("&&&")[0], "MOV")
                            else:
                                text = re.sub(u"\[\[\[" + url_name + ".*?\]\]\]", url_name.split("&&&")[1], text)
                        else:
                            text = re.sub(u"\[\[\[" + url_name + ".*?\]\]\]", url_name.split("&&&")[1], text)
                    else:
                        text = re.sub(u"\[\[\["+ url_name + ".*?\]\]\]",url_name.split("&&&")[1],text)
                info = {}
                info["url"] = url
                info["text"] = text
                self.mongodb.nerTrainSet.insert(info)
        except (Exception) as e:
            print (url)
            print (e)

    def jieba_utils(self):
        file = codecs.open("word2vec/actor_baike.txt","wb","utf-8")
        objects = self.mongodb.nerTrainSet.find({}).batch_size(50)
        for object in objects:
            try:
                if "text" in object.keys():
                    text = object["text"]
                    entity_list = re.findall(u'\[\[\[(.*?)\]\]\]',text)
                    if len(entity_list) > 0:
                        for entity in entity_list:
                            tag = entity.split("&&&")[0]
                            name = entity.split("&&&")[1]
                            if tag == "PER":
                                jieba.suggest_freq(name, True)
                            text = re.sub(u"\[\[\[" + entity + ".*?\]\]\]", name, text)
                    file.write(' '.join(jieba.cut(text, cut_all=False)))
            except Exception as e:
                print (object["url"])
                print (e)
        file.close()

    def set_tag_bio(self,text, tag):
        txt = ""
        if tag == "O":
            segment = jieba.posseg.cut(text)
            for seg in segment:
                txt += seg.word + "\t" + seg.flag + "\t" + "O" + "\n"
        else:
            segment = list(jieba.posseg.cut(text))
            txt += segment[0].word + "\t" + segment[0].flag + "\t" + "B-" + tag[0:3] + "\n"
            for i in range(1, len(segment)):
                txt += segment[i].word + "\t" + segment[i].flag + "\t" + "I-" + tag[0:3] + "\n"
        return txt

    def tag_txt_ner(self,str):
        text = ""
        strlist = re.split(u"\[\[\[|\]\]\]", str)
        for txt in strlist:
            if len(re.findall(u"&&&", txt)):
                text += self.set_tag_bio(txt.split("&&&")[1], txt.split("&&&")[0])
            else:
                text += self.set_tag_bio(txt, "O")
        return text

    def getData(self):
        trainfile = codecs.open("./trainset.txt","wb","utf-8")
        testfile = codecs.open("./testset.txt","wb","utf-8")
        objects = self.mongodb.nerTrainSet2.find({}).batch_size(50)
        i = 0
        for object in objects:
            try:
                if "text" in object.keys():
                    text = object["text"]
                    text = self.tag_txt_ner(text)
                    if i < 2000:
                        i += 1
                        trainfile.write(text)
                        continue
                    if i < 2400:
                        i += 1
                        testfile.write(text)
                        continue
                    break
            except Exception as e:
                print (object["url"])
                print (e)
        trainfile.close()
        testfile.close()


if __name__ == "__main__":
    ner = Ner()
    # ner.jieba_utils()
    ner.getData()
    # print("MOV%E5%95%8A"[0:3])