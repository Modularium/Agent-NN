import { IExecuteFunctions } from 'n8n-core';
import { IDataObject } from 'n8n-workflow';
import axios, { AxiosRequestConfig } from 'axios';

export async function execute(this: IExecuteFunctions): Promise<IDataObject[]> {
  const endpoint = this.getNodeParameter('endpoint') as string;
  const taskType = this.getNodeParameter('taskType') as string;
  const payload = this.getNodeParameter('payload') as IDataObject;
  const method = (this.getNodeParameter('method', 0, 'POST') as string).toUpperCase();
  const headers = (this.getNodeParameter('headers', 0, {}) as IDataObject) as Record<string, string>;
  const timeout = this.getNodeParameter('timeout', 0, 10000) as number;

  const options: AxiosRequestConfig = {
    method,
    url: `${endpoint}/task`,
    data: {
      task_type: taskType,
      input: payload,
    },
    headers,
    timeout,
  };

  const { data } = await axios.request(options);

  return [data as IDataObject];
}
