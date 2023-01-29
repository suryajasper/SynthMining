
function fetchRequest(endpoint, { method, body, query }) {
  return new Promise((resolve, reject) => {

    method = method.toUpperCase();

    const options = {
      method: method.toUpperCase(),
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
    };

    if (method === 'POST')
      options.body = JSON.stringify(body || query);
    else if (method === 'GET')
      options.query = JSON.stringify(query || body);

    fetch(`http://localhost:2002${endpoint}`, options)
      .then(res => res.json())
      .then(resolve)
      .catch(reject)
    
  })
}

export { fetchRequest };