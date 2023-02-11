const Cookies = {
  set(name : string, value : string, hours? : number) : void {
    let expr : string = '';

    if (hours) {
      const now : Date = new Date();
      const exprDate : Date = new Date();

      exprDate.setTime(now.getTime() + hours*60*60*1000);
      expr = exprDate.toUTCString();
    }
    
    document.cookie = `${name}=${value}; expires=${expr};"`;
  },

  get(name : string) : string | undefined {
    const cookieAttrs : string[] = document.cookie.split(';');

    for (let cookie of cookieAttrs) {
      cookie = cookie.trim();
      if (cookie.startsWith(`${name}=`)) 
        return cookie.substring(name.length+1);
    }
    
    return undefined;
  },

  erase(name: string) : void {
    document.cookie = name + '=; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
  },
}

export default Cookies;