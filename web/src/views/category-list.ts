import m from 'mithril';
import '../css/tags.scss';
import { fetchRequest, initArr } from '../utils/utils';
import { icons } from './icons';
import { TagAttrs } from './project-loader';

interface CategoryViewAttrs {
  tagName: TagAttrs;
  selected: boolean;
  color: string;
  select: () => void;
  activatePopup: () => void;
}

const CategoryView : m.Component<CategoryViewAttrs> = {
  view({attrs}) {
    const tag = attrs.tagName;

    return m('div.category-item', {
      class: attrs.selected ? 'selected' : '',
      onclick: attrs.select,
    }, [
      m('div.category-header', [
        m('span.category-item-color', {
          style: {
            backgroundColor: attrs.color,
          }
        }),
        m('span.category-item-title', tag.name),
      ]),
      
      m('div.category-description', tag.description),

      m('div.hover-menu', 
        m('button.tag-edit-button', {
          onclick: e => {
            e.stopPropagation();

            attrs.activatePopup();
          }
        }, 
          icons.gear,
        )
      )
    ]);
  }
};

export interface CategorySelectionsAttrs {
  uid: string | undefined;
  projectId: string;

  categories: TagAttrs[];
  colors: string[];

  addTag() : void;
  res(tag: TagAttrs) : void;
  updatePopupStatus(state: {
    name: string,
    active: boolean,
    data: object,
  }) : void;
  unshiftSelection(i: number);
}

export class CategorySelections implements m.ClassComponent<CategorySelectionsAttrs> {
  private selected: boolean[];

  constructor(vnode: m.CVnode<CategorySelectionsAttrs>) {
    this.selected = [];
  }

  createCategoryItem(
    vnode: m.CVnode<CategorySelectionsAttrs>,
    i: number,
  ) : m.Vnode<CategoryViewAttrs> {
    const tag = vnode.attrs.categories[i];

    return m(CategoryView, {
      tagName: tag,
      selected: this.selected[i],
      color: vnode.attrs.colors[i],

      select: () => {
        if (!this.selected[i]) {
          vnode.attrs.unshiftSelection(i);
          for (let j = this.selected.length-1; j >= 1; j--)
            this.selected[j] = this.selected[j-1];
          this.selected[0] = true;
        } else {
          this.selected[i] = false;
        }
      },

      activatePopup: () => {
        vnode.attrs.updatePopupStatus({
          name: 'editTag', 
          active: true,
          data: {
            projectId: vnode.attrs.projectId,
            uid: vnode.attrs.uid,
            tag,
          },
        });
      },
    })
  }

  view(vnode: m.CVnode<CategorySelectionsAttrs>) {
    if (this.selected.length != vnode.attrs.categories.length)
      this.selected = initArr<boolean>(vnode.attrs.categories.length, false);
    
    return m('div.category-container', [
      m('div.category-content',
        (vnode.attrs.categories || [])
          .map((cat, i) => 
            this.createCategoryItem(vnode, i)
          )
          .concat([
            m('div.category-item.new-category-button.selected', {
              onclick: vnode.attrs.addTag,
            }, [
              m('span.category-item-new-icon', '+'),
              m('span.category-item-title', 'Add Tag'),
            ])
          ])
      ),
    ]);
  }
}