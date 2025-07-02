import { IExecuteFunctions } from 'n8n-core';
import { IDataObject } from 'n8n-workflow';
import axios from 'axios';

export async function execute(this: IExecuteFunctions): Promise<IDataObject[]> {
  const endpoint = this.getNodeParameter('endpoint') as string;
  const taskType = this.getNodeParameter('taskType') as string;
  const payload = this.getNodeParameter('payload') as IDataObject;

  const { data } = await axios.post(`${endpoint}/task`, {
    task_type: taskType,
    input: payload,
  });

  return [data as IDataObject];
}
