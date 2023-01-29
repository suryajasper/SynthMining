import m from 'mithril';

function base64ImgHeader(string) {
  return `data:image/jpg;base64,${string}`
}

class ImageView {
  view(vnode) {
    return m('div.img-view-container', [
      m('div.img-view-content', [
        m('img.image-display', { src: base64ImgHeader(vnode.attrs.src) }),
        m('span.image-title', vnode.attrs.name),
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