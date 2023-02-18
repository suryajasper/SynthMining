import m from 'mithril';
import { ImageAttrs, TagAttrs } from './project-loader';
import { icons } from './icons';
import { initArr } from '../utils/utils';

const base64ImgHeader = (imgData: string) => `data:image/jpg;base64,${imgData}`;

interface ISharedImageAttrs {
  tagById: Record<string, TagAttrs>;
}

interface ImageDisplayAttrs extends ImageAttrs, ISharedImageAttrs {
  selected: boolean;
  obscured: boolean;
  select: (shiftKey?: boolean) => void;
}

class ImageView implements m.ClassComponent<ImageDisplayAttrs> {
  trimText(str : string) : string {
    let maxLen : number = 1000;

    if (str.length > maxLen)
      return str.substring(0, 10) + '...';

    return str;
  }

  view({attrs} : m.CVnode<ImageDisplayAttrs>) {
    return m('div.img-view-container', {
      class: [
        attrs.selected ? 'selected' : '',
        attrs.obscured ? 'not-selected' : '',
      ].join(' '),

      onclick: (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();

        attrs.select(e.shiftKey);
      },

      onmouseover: (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
      }
    }, [
      m('div.img-view-content', {
        title: attrs.name,
      }, [
        m('img.image-display', { src: base64ImgHeader(attrs.src) }),
        m('span.image-title', this.trimText(attrs.name)),
        m('div.image-tag-list', attrs.tags.map(tag => 
          m('span.image-tag-color', {
            style: { backgroundColor: attrs.tagById[tag].color },
          })
        ))
      ])
    ])
  }
}

const IconButton: m.Component<{
  title: string,
  icon: m.Child,
  onclick?: (e: MouseEvent) => void,
}> = {
  view({attrs}) {
    return m('button.button', {
      onclick: (e: MouseEvent) => {
        e.stopPropagation();
        attrs.onclick(e);
      },
    }, [
      attrs.icon,
      m('span', attrs.title),
    ]);
  }
}

interface ImageListAttrs extends ISharedImageAttrs {
  images: Array<ImageAttrs>;
  
  removeImages: (imageIds: string[]) => Promise<any>;
  changeImageSelection: (selectedIds: string[]) => void;
}

export default class ImageList implements m.ClassComponent<ImageListAttrs>
{
  private selection: {
    selected: boolean[],
    last: number;
  };
  private images: ImageAttrs[];
  private changeImageSelection: (selectedIds: string[]) => void;

  constructor({attrs}: m.CVnode<ImageListAttrs>) {
    this.selection = {
      selected: initArr<boolean>(attrs.images.length, false),
      last: 0,
    };
    this.images = [];
    this.changeImageSelection = attrs.changeImageSelection;
  }

  onupdate(vnode: m.CVnodeDOM<ImageListAttrs>) {
    if (this.images.length !== vnode.attrs.images.length)
      console.log('thre was an image update');
    this.images = vnode.attrs.images || [];

    if (this.images.length != this.selection.selected.length)
      this.selection.selected = initArr<boolean>(this.images.length, false);
  }

  clearSelection() : void {
    for (let i = 0; i < this.selection.selected.length; i++)
      this.selection.selected[i] = false;
    this.changeImageSelection(this.selectedIds);
  }

  selectAll() : void {
    for (let i = 0; i < this.selection.selected.length; i++)
      this.selection.selected[i] = true;
      this.changeImageSelection(this.selectedIds);
  }

  select(i: number, shiftKey?: boolean) : void {
    let newVal : boolean = !this.selection.selected[i]

    this.selection.selected[i] = newVal;
    if (shiftKey) {
      for (let j = this.selection.last; j < i; j++)
        this.selection.selected[j] = newVal;
    }

    this.selection.last = i;

    this.changeImageSelection(this.selectedIds);
  }

  get selectedIds() {
    return this.images
      .map(img => img._id)
      .filter((_, i) => this.selection.selected[i]);
  }

  get selectedCount() {
    let count = 0;
    for (let val of this.selection.selected)
      if (val) count++;
    return count;
  }
  
  view({attrs} : m.CVnode<ImageListAttrs>) {
    return [
      m('div.img-list-header', {
        class: this.selectedCount > 0 ? 'active' : '',
      }, [
        m(IconButton, {
          title: 'Dispatch to GAN',
          icon: icons.build,
        }),
        m(IconButton, {
          title: 'Select All',
          icon: icons.cloud,
          onclick: () => {
            this.selectAll();
          },
        }),
        m(IconButton, {
          title: 'Clear Selection',
          icon: icons.exit,
          onclick: () => {
            this.clearSelection();
          },
        }),
        m(IconButton, {
          title: 'Delete Images',
          icon: icons.trash,
          onclick: e => {
            attrs.removeImages(this.selectedIds);
            this.clearSelection();
          },
        }),
      ]),

      m('div.img-list-container',
        this.images.map(
          (img, i) => m(ImageView, Object.assign(img, {
            select: (shiftKey: boolean) => {
              this.select(i, shiftKey);
            },
            selected: this.selection.selected[i],
            obscured: this.selectedCount > 0 && !this.selection.selected[i],
            tagById: attrs.tagById,
          }))
        )
      )
    ]
  }
}