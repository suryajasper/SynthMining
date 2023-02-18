import m from 'mithril';
import '../css/tags.scss';
import { fetchRequest, initArr } from '../utils/utils';
import { icons } from './icons';
import { TagAttrs } from './project-loader';

interface CategoryViewAttrs {
  tagName: TagAttrs;
  selected: boolean;
  color: string;
  applyToImageMode: boolean;

  select: () => void;
  applyToImage: () => void;
  activatePopup: () => void;
}

const CategoryView : m.Component<CategoryViewAttrs> = {
  view({attrs}) {
    const tag = attrs.tagName;

    return m('div.category-item', {
      class: (attrs.selected || attrs.applyToImageMode) ? 'selected' : '',
      onclick: (e: MouseEvent) => {
        if (attrs.applyToImageMode)
          attrs.applyToImage();
        else
          attrs.select();
      },
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

      attrs.applyToImageMode ? null : 
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

  addTag: () => void;
  updatePopupStatus: (state: {
    name: string,
    active: boolean,
    data: object,
  }) => void;
  unshiftSelection: (i: number) => void;
  applyToImage: (tag: TagAttrs) => void;

  applyToImageMode: boolean;
}

export class CategorySelections implements m.ClassComponent<CategorySelectionsAttrs> {
  private selected: boolean[];

  constructor(vnode: m.CVnode<CategorySelectionsAttrs>) {
    this.selected = [];
  }

  createCategoryItem(
    attrs: CategorySelectionsAttrs,
    i: number,
    applyToImageMode: boolean,
  ) : m.Vnode<CategoryViewAttrs> {
    const tag = attrs.categories[i];

    return m(CategoryView, {
      tagName: tag,
      selected: this.selected[i],
      color: attrs.categories[i].color,
      applyToImageMode,

      select: () => {
        if (!this.selected[i]) {
          attrs.unshiftSelection(i);
          for (let j = this.selected.length-1; j >= 1; j--)
            this.selected[j] = this.selected[j-1];
          this.selected[0] = true;
        } else {
          this.selected[i] = false;
        }
      },

      applyToImage: () => {
        attrs.applyToImage(attrs.categories[i]);
      },

      activatePopup: () => {
        attrs.updatePopupStatus({
          name: 'editTag', 
          active: true,
          data: {
            projectId: attrs.projectId,
            uid: attrs.uid,
            tag,
          },
        });
      },
    })
  }

  view({attrs}: m.CVnode<CategorySelectionsAttrs>) {
    if (this.selected.length != attrs.categories.length)
      this.selected = initArr<boolean>(attrs.categories.length, false);
    
    return m('div.category-container', [
      attrs.applyToImageMode ?
        m('span.category-list-header', 'Click Tags to Apply to Images') : null,

      m('div.category-content',
        (attrs.categories || [])
          .map((_, i) => 
            this.createCategoryItem(attrs, i, attrs.applyToImageMode)
          )
          .concat(attrs.applyToImageMode ? 
            [] : [
              m('div.category-item.new-category-button.selected', {
                onclick: attrs.addTag,
              }, [
                m('span.category-item-new-icon', '+'),
                m('span.category-item-title', 'Add Tag'),
              ])
            ]
          )
      ),
    ]);
  }
}