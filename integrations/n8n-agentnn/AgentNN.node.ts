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
import { IExecuteFunctions } from 'n8n-core';
import {
  IDataObject,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';
import axios, { AxiosRequestConfig } from 'axios';

export class AgentNN implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'AgentNN',
    name: 'agentnn',
    group: ['transform'],
    version: 1,
    description: 'Send tasks to the Agent-NN dispatcher',
    defaults: {
      name: 'AgentNN',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Endpoint',
        name: 'endpoint',
        type: 'string',
        default: 'http://localhost:8000',
      },
      {
        displayName: 'Task Type',
        name: 'taskType',
        type: 'string',
        default: 'chat',
      },
      {
        displayName: 'Payload',
        name: 'payload',
        type: 'json',
        default: '{}',
      },
      {
        displayName: 'Path',
        name: 'path',
        type: 'string',
        default: '/task',
        description: 'API path appended to the endpoint',
      },
      {
        displayName: 'Method',
        name: 'method',
        type: 'options',
        options: [
          {
            name: 'POST',
            value: 'POST',
          },
          {
            name: 'GET',
            value: 'GET',
          },
        ],
        default: 'POST',
        description: 'HTTP method to use',
      },
      {
        displayName: 'Headers',
        name: 'headers',
        type: 'json',
        default: '{}',
        description: 'Optional HTTP headers',
      },
      {
        displayName: 'Timeout',
        name: 'timeout',
        type: 'number',
        default: 10000,
        description: 'Request timeout in milliseconds',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const endpoint = this.getNodeParameter('endpoint') as string;
    const taskType = this.getNodeParameter('taskType') as string;
    const payload = this.getNodeParameter('payload') as IDataObject;
    const path = this.getNodeParameter('path', 0, '/task') as string;
    const method = (this.getNodeParameter('method', 0, 'POST') as string).toUpperCase();
    const headers = (this.getNodeParameter('headers', 0, {}) as IDataObject) as Record<string, string>;
    const timeout = this.getNodeParameter('timeout', 0, 10000) as number;

    const options: AxiosRequestConfig = {
      method,
      url: `${endpoint}${path}`,
      data: {
        task_type: taskType,
        input: payload,
      },
      headers,
      timeout,
    };

    const { data } = await axios.request(options);

    return [[data as INodeExecutionData]];
  }
}
