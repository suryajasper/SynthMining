import m from 'mithril';
import { ImageAttrs, TagAttrs } from './project-loader';
import { icons } from './icons';
import { initArr } from '../utils/utils';

const base64ImgHeader = (imgData: string) => `data:image/jpg;base64,${imgData}`;

interface ISharedImageAttrs {
  tagById: Record<string, TagAttrs>;
  changeTagHighlight: (highlightedTagIds: string[]) => void; 
}

interface ImageDisplayAttrs extends ImageAttrs, ISharedImageAttrs {
  selected: boolean;
  obscured: boolean;
  select: (shiftKey?: boolean) => void;
}

interface ImageListAttrs extends ISharedImageAttrs {
  uid: string;

  images: Array<ImageAttrs>;
  isAdmin: boolean;
  
  removeImages: (imageIds: string[]) => Promise<any>;
  setValidationStatus: (imageIds: string[], validate: boolean) => void;
  changeImageSelection: (selectedIds: string[]) => void;
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
        attrs.validated ? 'img-validated' : '',
        attrs.selected ? 'selected' : '',
        attrs.obscured ? 'not-selected' : '',
      ].join(' '),

      onclick: (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();

        attrs.select(e.shiftKey);
      },

      onmouseenter: (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();

        attrs.changeTagHighlight(attrs.tags);
      },

      onmouseleave: (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();

        attrs.changeTagHighlight([]);
      },
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

export default class ImageList implements m.ClassComponent<ImageListAttrs>
{
  private uid: string;
  private isAdmin: boolean;

  private selection: {
    selected: boolean[],
    last: number;
  };
  private images: ImageAttrs[];
  private changeImageSelection: (selectedIds: string[]) => void;

  constructor({attrs}: m.CVnode<ImageListAttrs>) {
    this.uid = attrs.uid;
    this.selection = {
      selected: initArr<boolean>(attrs.images.length, false),
      last: 0,
    };
    this.images = [];
    this.changeImageSelection = attrs.changeImageSelection;
  }

  onupdate(vnode: m.CVnodeDOM<ImageListAttrs>) {
    this.isAdmin = vnode.attrs.isAdmin;

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

  canSelect(i) : boolean {
    return this.isAdmin || (
      !this.images[i].validated && 
      this.images[i].authorId === this.uid
    );
  }

  select(i: number, shiftKey?: boolean) : void {
    let newVal : boolean = !this.selection.selected[i]

    if (this.canSelect(i))
      this.selection.selected[i] = newVal;
    
    if (shiftKey) {
      for (let j = this.selection.last; j < i; j++) {
        if (this.canSelect(j))
          this.selection.selected[j] = newVal;
        else
          this.selection.selected[j] = false;
      }
    }

    this.selection.last = i;

    this.changeImageSelection(this.selectedIds);
  }

  createImageView(attrs: ImageListAttrs, img, i) : m.Child {
    return m(ImageView, Object.assign(img, {
      select: (shiftKey: boolean) => {
        this.select(i, shiftKey);
      },
      selected: this.selection.selected[i],
      obscured: this.selectedCount > 0 && !this.selection.selected[i],

      tagById: attrs.tagById,
      changeTagHighlight: attrs.changeTagHighlight,
    }));
  }

  get selectedIds() : string[] {
    return this.images
      .map(img => img._id)
      .filter((_, i) => this.selection.selected[i]);
  }

  get selectedCount() : number {
    let count = 0;
    for (let val of this.selection.selected)
      if (val) count++;
    return count;
  }

  get isSelectionInvalidated() : boolean {
    for (let i = 0; i < this.selection.selected.length; i++) {
      if (this.selection.selected[i] && this.images[i].validated)
        return false;
    }
    return true;
  }
  
  view({attrs} : m.CVnode<ImageListAttrs>) {
    let i = 0;

    return [
      m('div.img-list-header', {
        class: this.selectedCount > 0 ? 'active' : '',
      }, [
        m(IconButton, {
          title: 'Clear Selection',
          icon: icons.exit,
          onclick: () => {
            this.clearSelection();
          },
        }),

        this.isSelectionInvalidated ? 
        [
          m(IconButton, {
            title: 'Validate',
            icon: icons.subscription.bunny,
            onclick: e => {
              attrs.setValidationStatus(this.selectedIds, true);
              this.clearSelection();
            }
          }),
          m(IconButton, {
            title: 'Reject',
            icon: icons.trash,
            onclick: e => {
              attrs.removeImages(this.selectedIds);
              this.clearSelection();
            },
          }),
        ] :
        [
          m(IconButton, {
            title: 'Dispatch to GAN',
            icon: icons.build,
          }),
          m(IconButton, {
            title: 'Delete Images',
            icon: icons.trash,
            onclick: e => {
              attrs.removeImages(this.selectedIds);
              this.clearSelection();
            },
          }),
        ]
      ]),

      m('div.img-list-container', [
        m('span.img-list-title', 'Unvalidated'),
        m('div.img-list', 
          this.images
            .filter(img => !img.validated)
            .map(img => this.createImageView(attrs, img, i++))
        ),
        m('span.img-list-title', 'Validated'),
        m('div.img-list', 
          this.images
            .filter(img => img.validated)
            .map(img => this.createImageView(attrs, img, i++))
        ),
      ])
    ]
  }
}