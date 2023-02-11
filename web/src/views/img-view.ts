import m from 'mithril';
import { ImageAttrs } from './project-loader';

const base64ImgHeader = (imgData: string) => `data:image/jpg;base64,${imgData}`;

class ImageView implements m.ClassComponent<ImageAttrs> {
  trimText(str : string) : string {
    let maxLen : number = 1000;

    if (str.length > maxLen)
      return str.substring(0, 10) + '...';

    return str;
  }

  view({attrs} : m.CVnode<ImageAttrs>) {
    return m('div.img-view-container', {
      onclick: e => {
        e.preventDefault();
        e.stopPropagation();
      },
      onmouseover: e => {
        e.preventDefault();
        e.stopPropagation();
      }
    }, [
      m('div.img-view-content', {
        title: attrs.name,
      }, [
        m('img.image-display', { src: base64ImgHeader(attrs.src) }),
        m('span.image-title', this.trimText(attrs.name)),
      ])
    ])
  }
}

export default class ImageList 
  implements m.ClassComponent<{
    images: Array<ImageAttrs>
  }>
{
  view({attrs} : m.CVnode<{images: Array<ImageAttrs>}>) {
    return m('div.img-list-container',
      attrs.images.map(
        img => m(ImageView, img)
      )
    );
  }
}