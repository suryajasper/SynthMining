import m from 'mithril';

function base64ImgHeader(string) {
  return `data:image/jpg;base64,${string}`
}

class ImageView {
  trimText(str) {
    let maxLen = 1000;

    if (str.length > maxLen)
      return str.substring(0, 10) + '...';

    return str;
  }

  view(vnode) {
    return m('div.img-view-container', [
      m('div.img-view-content', {
        title: vnode.attrs.name,
      }, [
        m('img.image-display', { src: base64ImgHeader(vnode.attrs.src) }),
        m('span.image-title', this.trimText(vnode.attrs.name)),
      ])
    ])
  }
}

export default class ImageList {
  view(vnode) {
    return m('div.img-list-container', 
      vnode.attrs.images.map(
        img => m(ImageView, img)
      )
    );
  }
}