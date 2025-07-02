import axios from 'axios';

export async function execute(this: any): Promise<any[]> {
  const endpoint = this.getNodeParameter('endpoint') as string;
  const taskType = this.getNodeParameter('taskType') as string;
  const payload = this.getNodeParameter('payload') as any;
  const headers = (this.getNodeParameter('headers', 0, {}) as any) || {};
  const method = (this.getNodeParameter('method', 0, 'POST') as string).toUpperCase();
  const timeout = this.getNodeParameter('timeout', 0, 10000) as number;

  const { data } = await axios.request({
    url: `${endpoint}/task`,
    method,
    data: {
      task_type: taskType,
      input: payload,
    },
    headers,
    timeout,
  });

  return [data as any];
}
