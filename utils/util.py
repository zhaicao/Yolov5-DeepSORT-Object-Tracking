import os
import yaml
import pymysql
import numpy as np


def readConfigs(filename="configs.yaml"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} is not found")
    with open(filename, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f.read())
        return configs

def initFlowMetrix(obj_list):
    """
    初始化流量矩阵
    for example:
            [["类型", "Car", "BUS", "Truck", "Total"],
            ["北向流量", "0", "0", "0", "0"],
            ["南向流量", "0", "0", "0", "0"]])
    """
    obj_list = obj_list + ["Total"]
    flow_metrix = np.zeros((2, len(obj_list)), dtype=np.int32)
    first_row = np.array([obj_list])
    first_col = np.array([["类型", "北向流量", "南向流量"]]).transpose()
    flow_metrix = np.concatenate((first_row, flow_metrix), axis=0)
    flow_metrix = np.concatenate((first_col, flow_metrix), axis=1)
    return flow_metrix

def updateFlowMetrix(flow_metrix, head, direction, num):
    """
    更新流量矩阵中数据
    """
    indices = np.where(flow_metrix == head)
    row_index = 1
    if direction == "south":
        row_index = 2
    flow_metrix[row_index, indices[1]] = int(flow_metrix[row_index, indices[1]]) + num  # 更新该类别数据
    flow_metrix[row_index, -1] = int(flow_metrix[row_index, -1]) + num  # 更新总数
    return flow_metrix


class MysqlOperation(object):

    def __init__(self):
        conf = readConfigs()
        self.host = conf["database"]["mysql"]["host"]
        self.user = conf["database"]["mysql"]["user"]
        self.password = conf["database"]["mysql"]["password"]
        self.port = conf["database"]["mysql"]["port"]
        self.dbname = conf["database"]["mysql"]["db"]
        self.conn = None  # 连接
        self.cur = None  # 游标

    def open(self):
        # 创建连接
        self.conn = pymysql.connect(host=self.host, user=self.user, password=self.password, port=self.port,
                                    db=self.dbname, charset='utf8')  # 创建连接
        self.cur = self.conn.cursor()  # 创建游标

    def select(self, sql):
        '''查询数据'''
        self.cur.execute(sql)  # 查询数据
        return self.cur.fetchall()  # 获取结果

    def execute(self, sql):
        '''执行sql'''
        try:
            # 执行SQL语句
            self.cur.execute(sql)
            # 提交事务到数据库执行
            self.conn.commit()  # 事务是访问和更新数据库的一个程序执行单元

        except BaseException as f:
            print(f)
            self.conn.rollback()

        # 返回受影响行数
        return self.cur.rowcount

    def executemany(self, sql, params):
        '''
        批量插入数据
        :param sql:    插入数据模版, 需要指定列和可替换字符串个数
        :param params:  插入所需数据，列表嵌套元组[(1, '张三', '男'),(2, '李四', '女'),]
        :return:    影响行数
        '''
        try:
            # sql = "INSERT INTO USER VALUES (%s,%s,%s,%s)"  # insert 模版
            # params = [(2, 'fighter01', 'admin', 'sanpang'),
            #           (3, 'fighter02', 'admin', 'sanpang')]  # insert数据，
            self.cur.executemany(sql, params)

        except BaseException as f:
            print(f)
            self.conn.rollback()

        return self.cur.rowcount

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''退出时关闭游标关闭连接'''
        self.cur.close()
        self.conn.close()
