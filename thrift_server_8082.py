from summarization import SummarizationModel
from summarization import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import json
import os
import subprocess

HOST = '192.168.101.11'
PORT = 8082

DATA_PATH = '/data/topic-aware-pointer-model/rpc_log'
INPUT_FILE = 'tmp_input_%i.txt' % PORT
OUTPUT_FILE = 'tmp_output_%i.txt' % PORT


def write_to_input_file(article_list):
  with open(os.path.join(DATA_PATH, INPUT_FILE), "w", encoding='utf-8') as writer:
    for article in article_list:
      writer.write("%s\n" % json.dumps({"text": article["text"], "id": article["id"]}, ensure_ascii=False))


def read_from_output_file():
  results = []
  with open(os.path.join(DATA_PATH, OUTPUT_FILE), "r", encoding='utf-8') as lines:
    for line in lines:
      line = line.strip()
      if line:
        results.append(json.loads(line))
  return results
  


class SummarizationHandler(object):
  def predict(self, articles):
    article_list = [{"id": a.id, "text": a.text} for a in articles]

    write_to_input_file(article_list)

    try:
      subprocess.check_call(["python", "run_summarization.py", "--data_path=%s" % os.path.join(DATA_PATH, INPUT_FILE)])
    except subprocess.CalledProcessError:
      return []

    decode_results = read_from_output_file()

    return [ttypes.Summarization(r["id"], r["summarization"]) for r in decode_results]


if __name__ == '__main__':
  handler = SummarizationHandler()

  processor = SummarizationModel.Processor(handler)
  transport = TSocket.TServerSocket(HOST, PORT)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()

  rpcServer = TServer.TSimpleServer(processor,transport, tfactory, pfactory)

  print('Starting the rpc server at', HOST,':', PORT)
  rpcServer.serve()