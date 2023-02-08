export function initArr(size, val=0) {
  let arr = [];
  for (let i = 0; i < size; i++)
    arr.push(val);
  return arr;
}

export function fetchRequest(endpoint, { method, body, query }) {
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
      options.body = JSON.stringify(body || query);
    else
      url += '?' + new URLSearchParams(query || body);

    
    fetch(url, options)
      .then(res => res.json())
      .then(resolve)
      .catch(reject)
    
  })
}

export function randId(length) {
  let result = '';
  const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const charactersLength = characters.length;
  let counter = 0;
  while (counter < length) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
    counter += 1;
  }
  return result;
}