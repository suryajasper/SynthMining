const Cookies = {
  set(name, value, hours) {
    let expr = '';

    if (hours) {
      const now = new Date();
      const exprDate = new Date();
      exprDate.setTime(now.getTime() + hours*60*60*1000);
      expr = exprDate.toUTCString();
    }
    
    document.cookie = `${name}=${value}; expires=${expr};"`;
  },

  get(name) {
    const cookieAttrs = document.cookie.split(';');
    console.log(cookieAttrs);

    for (let cookie of cookieAttrs) {
      cookie = cookie.trim();
      if (cookie.startsWith(`${name}=`)) 
        return cookie.substring(name.length+1);
    }
    
    return undefined;
  },

  erase(name) {
    document.cookie = name + '=; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
  },
}

export default Cookies;