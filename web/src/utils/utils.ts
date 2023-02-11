export function initArr<Type>(size: number, val:Type) : Type[] {
  let arr: Type[] = [];
  for (let i = 0; i < size; i++)
    arr.push(val);
  return arr;
}

interface RequestContent {
  method: string;
  body?: object;
  query?: Record<string, string>;
}

export function fetchRequest<ResponseType>(
  endpoint: string, requestData : RequestContent
) : Promise<ResponseType> {

  let { method, body, query } = requestData;

  return new Promise((resolve, reject) => {

    method = method.toUpperCase();

    const options = {
      method: method.toUpperCase(),
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
    };

    let url = `http://localhost:2002${endpoint}`;

    if (method === 'POST')
      options['body'] = JSON.stringify(body || query);
    else
      url += '?' + new URLSearchParams(query);

    
    fetch(url, options)
      .then(res => res.json())
      .then(resolve)
      .catch(reject)
    
  })

}

export function randId(length: number) : string {
  let result : string = '';
  const characters : string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const charactersLength : number = characters.length;
  let counter : number = 0;
  while (counter < length) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
    counter += 1;
  }
  return result;
}

export type Modify<T, R> = Omit<T, keyof R> & R;